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

Versión 3.2.3:
- [x] Added Catalan translation.

- [ ] Added Italian translation.

