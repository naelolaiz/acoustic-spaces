# Outdoor scenes

Sound outdoors: a street, a square, a courtyard, a stage in a park, a noise barrier by a
road. A scene is outdoors when its geometry has a `ground`: a flat ground under an open
sky, with buildings, barriers and the objects of a model file standing on it.

```yaml
geometry:
  ground: {margin: 30}          # flat, at z = 0, 30 m past everything on it
  buildings:
    - name: houses
      footprint: [[-30, 6], [10, 6], [10, 16], [-30, 16]]   # corners, in order, any shape
      height: 9
  barriers:
    - name: noise_barrier
      path: [[14, 6], [44, 6]]  # points along the ground
      height: 3
assignments:
  ground: asphalt
  houses: brick
  barriers: concrete
```

See [`examples/street/scene.yaml`](https://github.com/naelolaiz/acoustic-spaces/blob/main/examples/street/scene.yaml):
traffic along a street between two buildings, a balcony, and a garden behind a noise
barrier.

## The ground and the sky

The ground, four walls of sky round its edge and the sky overhead close a box, so the
open air is a room like any other: its volume is known, sources and listeners are
checked to stand in it, and the tracers know which side of every face the air is on.

- `ground.height` is the ground's height (z), 0 by default.
- `ground.extent: [[x_min, y_min], [x_max, y_max]]` is where the ground ends. Without
  it, the ground reaches `ground.margin` (30 m by default) past the buildings, barriers
  and the model file's objects. A scene with none of them needs an extent.
- `ground.sky` is how high the sky is over the ground; without it, `margin` over the
  highest building, barrier or object.

The sky is the material `sky`, an opening: everything that reaches it passes through
and leaves the scene, as sound leaves into the open. Nothing reflects off it, its edges
diffract nothing and its edge with the ground does not either, and the ground scatters
nothing for its size (as a surface of limited size does at low frequencies, see
[Scattering](simulation.md#scattering)), so the ground reads as if it went on for ever: a source over the ground has its direct sound and one reflection
off the ground, and nothing else. A scene may give the sky a material of its own
(`assignments: {sky: ...}`); otherwise the scene file leaves it out.

The ground takes a material as any surface does. A material with an `impedance` (grass,
soil, snow, gravel, asphalt) reflects sound as a real ground does, with the phase and
the ground wave that make the dip over soft ground (see [Ground impedance](#ground-impedance));
one with absorption coefficients alone reflects as a floor does. Parts of the ground
can have materials of their own: a road, a lawn, a car park (see
[Ground zones](#ground-zones)).

The ground and the sky are the surfaces `ground` and `sky`.

## Buildings and barriers

A building is its `footprint` (corners in order, as a floor plan) raised to its
`height` over the ground: walls and a flat roof, open underneath, where it stands on the
ground. Its surfaces are named as a floor plan room's walls are, after the building:

| Surface | What it is |
| --- | --- |
| `<name>_roof` | its roof |
| `<name>_wall_1`, `<name>_wall_2`, ... | the wall from the footprint's first corner to its second, and so on |
| `<name>_walls`, `<name>` | its walls; all of it |
| `walls`, `roofs`, `buildings` | those of every building |

A barrier is a thin screen along a `path` of two or more points, from the ground to its
`height`, reflecting on both sides: surfaces `<name>` and `barriers`. Buildings and
barriers without a `name` are `building_1`, `barrier_1` and so on. The names `ground`,
`sky`, `walls`, `roofs`, `buildings` and `barriers` are taken, and no two may share one.

A model file (`geometry.file`, with its `units` and `up_axis`) adds what it holds:
buildings of any shape, a stage, a bandshell, terrain features. Its objects stand on the
ground and must not reach below it; faces lying on the ground (a building's floor, the
file's own ground plane) are left out, so they are open underneath too. With an
`extent`, everything must lie inside it. The file's objects are taken as they are: a
point inside one is not reported, and the inside corners of a building drawn in the
file may diffract, which those of `buildings` do not; give buildings as `buildings`
where you can.

Sources and listeners must stand in the open air: above the ground, under the sky,
inside the extent and not inside one of the `buildings`; `validate` and the editor say
which one is not.

## Ground impedance

A material with an `impedance` is a porous ground. Its impedance follows from its flow
resistivity σ (kPa s/m², how hard it is to push air through it), by a preset or a value
of its own:

```yaml
materials:
  lawn: {impedance: grass}                    # a preset (see below)
  field:
    impedance: {flow_resistivity: 150, model: delany_bazley}
  snow_on_ice:
    impedance: {preset: snow, thickness: 0.1} # a layer 10 cm thick on a hard backing
  pasture:
    impedance: {model: two_parameter, flow_resistivity: 100, porosity_rate: 20}
assignments:
  ground: lawn
```

- `model` is the empirical model of porous media the impedance comes from: `miki` (Miki,
  1990; the default), which stays passive and causal at every frequency;
  `delany_bazley` (Delany and Bazley, 1970), as most of the outdoor literature quotes
  it; or `two_parameter` (Attenborough, 1985), a porosity that falls off with depth at
  `porosity_rate` (1/m) under an effective flow resistivity, which fits measured grass
  and soil better than one parameter does.
- `thickness` (m) makes the ground a layer of that thickness on a hard backing: snow on
  frozen ground, a thin soil over rock. Not with `two_parameter`.
- The presets are the ground classes of Nord2000, with their flow resistivities. Real
  grounds spread over a factor of two or more round them.

| `preset` | σ (kPa s/m²) | Ground |
| --- | ---: | --- |
| `snow` | 12.5 | very soft: new snow, moss |
| `forest_floor` | 31.5 | soft forest floor, short dense heather, thick moss |
| `loose_soil` | 80 | uncompacted loose ground: turf, loose soil |
| `grass` | 200 | normal uncompacted ground: pasture, lawns |
| `compacted_soil` | 500 | compacted field and gravel: trodden lawns, parks |
| `gravel` | 2000 | compacted dense ground: gravel roads, car parks |
| `asphalt` | 20000 | hard surfaces: most asphalt and concrete |
| `water` | 200000 | very hard and dense: dense asphalt, concrete, water |

A material with an impedance takes its absorption from it: the random-incidence
absorption of its impedance (Paris's formula), in third-octave bands, which the late
part's rays use and `inspect` shows (grass absorbs 0.18 at 63 Hz and 0.89 at 8 kHz). It
takes no `preset`, `file` or transmission, and any `absorption` given with it is
replaced; its `scattering` is its own.

The early paths use the impedance itself. A point source's spherical wave does not
reflect off a ground as a plane wave does: near grazing incidence, where the plane wave's
reflection factor R tends to -1 and would cancel the direct sound, a ground wave makes
up part of it. A reflection off a ground takes the spherical wave's factor Q = R +
(1 - R) F(w) (Weyl-van der Pol), with F the ground wave's, which depends on the path's
length, the angle of incidence and the frequency. It is the asymptotic form of the exact
solution for a locally reacting plane: from 5 m away, the level is within 0.05 dB of the
exact solution's over grass and harder grounds, 0.2 dB over snow and 0.4 dB over a
layer of snow 10 cm thick, more only in dips over 20 dB deep (see
[Validation](validation.md#ground-reflection-against-the-exact-solution)).

Over soft ground the reflection arrives turned in phase and cancels part of the direct
sound over a band of frequencies: the ground dip, at a few hundred hertz for a source and
a listener a metre or two up over grass, deeper and wider the farther apart they are.
Over hard ground the dip lies where the reflected path is half a wavelength longer,
often above 1 kHz. In the `roadside` example, 25 m from the road and 1.5 m up, the
500 Hz band is 10.7 dB quieter over the meadow than over the paved yard across the road,
and the yard has its own dip at 2 kHz.

So that the dip shows in every band, outdoors the early paths add in pressure in every
band by default (`simulation.early.coherent_max_band_hz: 8000`; a room takes 500 at
most, see [Low frequencies: early reflections in pressure](simulation.md#low-frequencies-early-reflections-in-pressure)):
outdoors the paths are few. Up to 500 Hz they go through the band filters as indoors;
higher up each pair of paths adds its cross term as far as the band leaves the two
coherent, which it does less the more their delays differ. The points of a line or
surface source still add in energy, as they are uncorrelated; the paths from each point
add in pressure.

The wave solver fits each ground to its impedance itself, its admittance and phase as
well as its absorption (see [the wave solver](simulation.md#low-frequencies-the-wave-solver)).
Any surface may have an impedance, indoors too. `calibrate` (see
[Fitting materials to measurements](measurements.md#fitting-materials-to-measurements))
fits absorption coefficients, so it leaves materials with an impedance alone, and
outdoor scenes, which do not decay as a room does.

## Ground zones

Zones are parts of the ground with a material of their own: a road across a meadow, a
lawn in a paved square, a car park.

```yaml
geometry:
  ground:
    margin: 30
    zones:
      - name: road
        polygon: [[-100, -4], [100, -4], [100, 4], [-100, 4]]   # corners, in order
      - name: car_park
        polygon: [[20, 4], [60, 4], [60, 40], [20, 40]]
assignments:
  ground: lawn        # the ground outside the zones, and zones without a material
  road: asphalt
  car_park: asphalt
```

A zone is a polygon on the ground, cut to the ground's extent; zones may touch but not
overlap. Its surfaces are `<name>`, `zones` and `ground`, so a zone without an
assignment of its own takes the ground's material. Zones without a `name` are `zone_1`,
`zone_2` and so on, and the names of zones, buildings and barriers are all different.

A reflection near where two grounds meet takes some of each. The sound that reflects
comes from a patch of ground round the reflection point, its first Fresnel zone: the
ellipse where a path through the ground is at most half a wavelength longer than the
reflected path, small at high frequencies and near normal incidence, long near grazing
incidence. The reflection factor is the sum of each ground's (Q for a ground with an
impedance, the square root of 1 - absorption for one without), each weighted by the
share of the Fresnel zone it covers, band by band, as engineering models such as
Nord2000 and Harmonoise weight mixed ground. Past the ground's extent the ground goes on
as it is at its edge.

## What the simulation does outdoors

- **Early paths** are found as in a room: the direct sound, reflections off the ground,
  façades and barriers, and (with `simulation.early.diffraction`) the sound bent over
  roofs, round the corners of buildings and over the top and round the ends of barriers.
  Edges on the ground (the foot of a wall or a barrier) and the inside corners where two
  wings of a building meet diffract nothing. `second_order` follows the sound over a
  building, its front and back edges in turn; `first_order` bends over one edge at a
  time, which is enough for a barrier.
- **The late part** comes from the stochastic ray tracer: rays reflect and scatter off
  the ground and the buildings until they leave through the sky. Between tall buildings
  (a street canyon, a courtyard) that is a tail of its own; in the open, almost nothing.
  The statistical late method needs a diffuse field, which the open air does not have:
  outdoors it stops with an error.
- **Response length.** Sources and listeners can stand hundreds of metres apart, so
  outdoors `ir_length_s` counts from the latest direct sound of a run: a listener 700 m
  away with `ir_length_s: 1` gets 3 s responses (2 s of travel, then 1 s). The run's
  details give the length (`ir_length_s`).
- **Coverage maps** cover the ground by default and leave out the points inside
  buildings (see [Coverage maps](coverage-maps.md)).
- **The wave solver** sees the sky as an opening and puts a matched layer round the
  grid, as for any open model.
- **Air** absorbs per band as indoors (ISO 9613-1, from `air`), along every path.
- **The early paths add in pressure in every band** (see
  [Ground impedance](#ground-impedance)): `simulation.early.coherent_max_band_hz` is
  8000 outdoors unless the scene sets it.

`acoustic-spaces inspect` says where the ground and the sky lie, and leaves out the
statistical reverberation times, which need a diffuse field.

## In the editor

The **Geometry** entry of an outdoor scene has an **Outdoors** group: the ground's height,
the margin and the sky, where the ground ends, the buildings and barriers with their
heights, which can be changed or removed, and the ground zones, which can be removed.
Footprints, paths and the zones' outlines are set in the scene file. A material's page
has a **Ground impedance** group: tick it to make the material a ground (grass to
start with), then pick the ground, the model, a flow resistivity of your own, a layer's
thickness or the porosity rate. Unticked, the material keeps the absorption the
impedance gave it. On the simulation page, **In pressure up to** goes up to 8000 Hz in an outdoor
scene. The 3D view and the export to Blender leave the sky out (the 3D view keeps its
edges, to show where the ground ends), and the status bar gives the ground's area
instead of a reverberation time. **Draw room** replaces an outdoor scene with a floor
plan after asking, and brings **In pressure up to** back to a room's default when it
was above 500 Hz.

## What is not modelled yet

These come next, each in its own step (see the [roadmap](roadmap.md)):

- wind and temperature gradients, which bend sound up (shadow zones) or down;
- turbulence, which scatters sound into shadow zones and blurs the ground dip;
- vegetation: trees, hedges and forests as volumes that scatter and absorb;
- the wave solver with a speed of sound that changes from place to place.

Until then the air is still and uniform and rays travel straight: results are good over
any ground at short and middle distances (up to a hundred metres or two, where wind and
temperature gradients bend sound little), and for street canyons and courtyards, where
the buildings matter most. The ground is flat and reacts locally: its impedance does not
change with the angle sound meets it at, which holds for grass, soil and snow, though
less for a thick layer of very soft snow.
