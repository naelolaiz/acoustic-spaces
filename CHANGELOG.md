# Changelog

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
