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
  against a separate particle simulation. Their late rays are traced on the CPU; the
  GPU devices may follow.
- [x] O6 The wave solver in wind and temperature: a speed of sound per cell (the
  effective one along the horizontal from the source) in the grid, its walls, its source
  and its matched layer; checked against the parabolic equation at 63 and 125 Hz. It
  runs on the CPU; the GPU may follow, and the flow itself (the linearised Euler
  equations) stays out.
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
  triangles. A ridge is within 0.2 dB of the exact wedge to 4 kHz, a flat terrain and a
  slope give the flat ground's and the image's paths, a smooth hill is within about a
  decibel of the thin screen's attenuation. Scenes without a terrain are unchanged.

Unreleased:

- [x] Editor: the background task in the status bar says what runs, with Cancel beside
  it; messages that can be shown or hidden by level, filtered by item or words, and
  cleared, every other one shaded.
