# Development

## Repository layout

```
src/acoustic_spaces/   Python package
  scene/               scene model and YAML schema
  geometry/            CAD and BIM import, triangulation, mesh cleanup and checks, the
                       rooms a model encloses (cells)
  materials/           material database, coefficient files, CATT-Acoustic and ODEON
                       material libraries
  directivity.py       source directivity: patterns per band, SOFA and CSV balloons
  room.py              a scene with its geometry loaded: materials per face, checks
  engine/              early reflections, edge diffraction, late reverberation (on the
                       CPU, or on a GPU through WebGPU), early paths in pressure,
                       grazing attenuation over the audience, the wave solver (CPU and
                       GPU) and its wall admittances, re-weighting for new absorption,
                       sound particles, the leak finder, responses, simulate(),
                       coverage maps
  air.py               ISO 9613-1 air absorption and speed of sound
  synthesis.py         octave filter bank and impulse response synthesis
  spatial/             Ambisonics, VBAP loudspeaker layouts, HRTFs, spatial rendering,
                       sound field rotation, binaural decoding of Ambisonics
  parameters.py        ISO 3382-1 parameters, octave filters, noise floor detection
  render.py            convolution, resampling, auralisation, real-time playback
  headtracking.py      head-tracked binaural playback, OSC head trackers
  measurement.py       sine sweeps, playback and recording, deconvolution
  compare.py           simulated against measured parameters, in JNDs (ISO 3382-1)
  calibration.py       absorption fitted to measured parameters (`calibrate`)
  walkthrough.py       listeners and sources heard as they walk through the room
  audio.py             WAV files
  viz.py               3D view, reflection paths, particle animations, response plots
  gltf.py              export to glTF (.glb) for Blender: room, paths, particle animation
  results.py           writing a simulation's responses and parameters to a folder
  examples.py          the example scenes that ship with the package
  gui/                 the graphical editor (Qt)
  cli.py               command-line interface
cpp/include/           header-only C++ core: BVH, ray tracer (specular paths, stochastic
                       and map tracing), wave solver (FDTD), FFT, partitioned
                       convolver (no Python, so a plugin or player can reuse it)
cpp/bindings/          nanobind module exposing the core as acoustic_spaces._native
tests/                 pytest suite, run on Linux, macOS and Windows in CI
examples/              example scenes and their scripts, installed with the package
validation/            the longer checks behind the validation report; bras/ has the
                       measured rooms of the BRAS benchmark
docs/                  the Sphinx documentation, the validation report and the pictures
```
