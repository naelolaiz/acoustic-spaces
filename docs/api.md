# API reference

The names most scripts need are also importable from the package itself:
`acoustic_spaces.Room`, `Scene`, `Source`, `Listener`, `Material` and `simulate`.

## Scenes and rooms

```{eval-rst}
.. automodule:: acoustic_spaces.scene
   :imported-members:
   :undoc-members:

.. automodule:: acoustic_spaces.room
```

## Source directivity

```{eval-rst}
.. automodule:: acoustic_spaces.directivity
```

## Geometry

```{eval-rst}
.. automodule:: acoustic_spaces.geometry
   :imported-members:
```

## Materials

```{eval-rst}
.. automodule:: acoustic_spaces.materials
   :imported-members:

.. automodule:: acoustic_spaces.materials.libraries
   :no-members:
```

## Simulation

```{eval-rst}
.. automodule:: acoustic_spaces.engine
   :imported-members:
```

## Coverage maps

{func}`acoustic_spaces.engine.coverage_map` and the classes it returns are listed under
Simulation above; the module also has the parameters a map shows and the A-weighting.

```{eval-rst}
.. automodule:: acoustic_spaces.engine.coverage
   :exclude-members: CoverageMap, CoverageResult, MapGrid, coverage_map, map_grid
```

## Edge diffraction

```{eval-rst}
.. automodule:: acoustic_spaces.engine.diffraction
```

## Early reflections in pressure

```{eval-rst}
.. automodule:: acoustic_spaces.engine.coherent
```

## Leak finder

```{eval-rst}
.. automodule:: acoustic_spaces.engine.leaks
```

## Audience seating

```{eval-rst}
.. automodule:: acoustic_spaces.engine.grazing
```

## Wave-based solver

```{eval-rst}
.. automodule:: acoustic_spaces.engine.wave
```

```{eval-rst}
.. automodule:: acoustic_spaces.engine.impedance
```

```{eval-rst}
.. automodule:: acoustic_spaces.engine.gpu
```

## Parameters, air and impulse responses

```{eval-rst}
.. automodule:: acoustic_spaces.parameters

.. automodule:: acoustic_spaces.air

.. automodule:: acoustic_spaces.synthesis
```

## Spatial outputs

```{eval-rst}
.. automodule:: acoustic_spaces.spatial
   :imported-members:
```

Rotating sound fields, and decoding them to the ears:

```{eval-rst}
.. automodule:: acoustic_spaces.spatial.rotation
.. automodule:: acoustic_spaces.spatial.decoder
```

## Rendering and playback

```{eval-rst}
.. automodule:: acoustic_spaces.render
```

## Walkthroughs

```{eval-rst}
.. automodule:: acoustic_spaces.walkthrough
```

## Head tracking

```{eval-rst}
.. automodule:: acoustic_spaces.headtracking
```

## Measurement

```{eval-rst}
.. automodule:: acoustic_spaces.measurement
```

## Comparing with measurements

```{eval-rst}
.. automodule:: acoustic_spaces.compare
```

## Fitting materials to measurements

```{eval-rst}
.. automodule:: acoustic_spaces.calibration
.. automodule:: acoustic_spaces.engine.reweight
```

## Audio files and results

```{eval-rst}
.. automodule:: acoustic_spaces.audio

.. automodule:: acoustic_spaces.results
```

## Example scenes

```{eval-rst}
.. automodule:: acoustic_spaces.examples
```

## 3D view and plots

```{eval-rst}
.. automodule:: acoustic_spaces.viz
```

## Export to Blender (glTF)

```{eval-rst}
.. automodule:: acoustic_spaces.gltf
```
