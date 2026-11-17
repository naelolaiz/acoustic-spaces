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
  products.py          matrix and dot products whose sums are added in a fixed order
  render.py            convolution, resampling, auralisation, real-time playback
  headtracking.py      head-tracked binaural playback, OSC head trackers
  measurement.py       sine sweeps, playback and recording, deconvolution
  compare.py           simulated against measured parameters, in JNDs (ISO 3382-1)
  calibration.py       absorption fitted to measured parameters (`calibrate`)
  walkthrough.py       listeners and sources heard as they walk through the room
  doppler.py           walks with Doppler: each early path through a changing delay
  audio.py             WAV files
  viz.py               3D view, reflection paths, particle animations, response plots
  gltf.py              export to glTF (.glb) for Blender: room, paths, particle animation
  results.py           writing a simulation's responses and parameters to a folder
  examples.py          the example scenes that ship with the package
  demos/               what the demos share: steps and options, recordings, mixes, the
                       tour, stills and video (a new demo describes its steps and options)
  gui/                 the graphical editor (Qt)
  cli.py               command-line interface
  i18n.py              the language of the text shown, and its translation
  locale/              the translation catalogs (es.po: Spanish, ca.po: Catalan,
                       it.po: Italian, de.po: German, nl.po: Dutch)
cpp/include/           header-only C++ core: BVH, ray tracer (specular paths, stochastic
                       and map tracing), wave solver (FDTD), FFT, partitioned
                       convolver, sums in a fixed order (no Python, so a plugin or
                       player can reuse it)
cpp/bindings/          nanobind module exposing the core as acoustic_spaces._native
tools/catalog.py       keeps the translation catalogs in step with the code (see Translating)
tests/                 pytest suite, run on Linux, macOS and Windows in CI
examples/              example scenes and their scripts, installed with the package
validation/            the longer checks behind the validation report; bras/ has the
                       measured rooms of the BRAS benchmark
docs/                  the Sphinx documentation, the validation report and the pictures
```

## Sums in a fixed order

numpy's `@` hands a product's sums to the linear-algebra library, which splits a large
one among its threads and adds the parts in an order that depends on how many it has
(as many as the computer's cores, unless `OPENBLAS_NUM_THREADS` says otherwise). The
last digits of such a sum can change with that number, and with them a run's results
from one computer to the next. So a product whose operands can grow with the scene (its
paths, frequencies, samples, receivers or faces) is written with
`acoustic_spaces.products`, whose compiled loops add each sum in the order its
definition fixes:

```python
from acoustic_spaces import products

spectra = products.product(weights.T, phased)  # weights.T @ phased
products.add_product(total, weights.T, phased)  # total += weights.T @ phased
norm = math.sqrt(products.dot(left, left))  # left @ left
```

A product of a few fixed values (3-vectors, 3 x 3 frames, a handful of bands) stays an
`@`: the library works it on one thread. The library's solvers (`eigh`, `pinv`, `svd`,
`inv`, `least_squares`) stay too, and so does live head-tracked playback.
`tests/test_code.py` counts the products each file leaves to the library and fails
when one is added, with a message that says what to do.
