# Outdoor scenes

Sound outdoors: a street, a square, a courtyard, a stage in a park, a noise barrier by a
road. A scene is outdoors when its geometry has a `ground`: a flat ground under an open
sky, with buildings, barriers, trees and hedges and the objects of a model file standing
on it.

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

## Wind and temperature

Outdoors the speed of sound changes with height. The air is warmer or cooler higher
up, and sound travels faster in warmer air; the wind, slowed by the ground, blows
faster higher up, and sound travels faster with it and slower against it. Sound bends
towards where it travels slower:

- **down** when its speed grows with height: downwind, and on a clear night, when the
  ground cools the air next to it (an inversion). Sound comes back to the ground again
  and again on its way, and carries further than in still air.
- **up** when its speed falls with height: upwind, and on a sunny day, when the ground
  warms the air next to it. Past some distance a listener near the ground is in a
  **shadow** that no ray reaches: only the sound that bends round its edge gets there,
  less and less with distance and with frequency.

```yaml
atmosphere:
  preset: windy                 # neutral, windy, sunny_day or clear_night
  wind: {direction_deg: 90}     # the preset's 5 m/s 10 m up, blowing towards +y
```

`atmosphere` is for outdoor scenes; in a room the air is still and one temperature, and
a room with an atmosphere stops with an error. Its fields:

| field | what it sets |
|---|---|
| `preset` | a weather to start from: `neutral` (still air), `windy` (5 m/s 10 m up), `sunny_day` (the air cooling with height: `temperature_log_k: -0.5`, `temperature_gradient_k_m: -0.0098`) or `clear_night` (the air warming with height: `temperature_log_k: 0.5`). Fields given with it change it. |
| `wind.speed_m_s`, `wind.height_m` | the wind's speed (m/s) at a height above the ground (m, 10 by default) |
| `wind.direction_deg` | the way the wind blows to, in degrees counterclockwise from +x seen from above (0: along +x, 90: along +y) |
| `wind.profile` | how its speed grows with height: `log`, as ln(1 + z / `roughness_m`), the usual profile over open ground (the default); `linear`, in proportion to the height; `constant` |
| `temperature_gradient_k_m` | how the temperature changes per metre of height (K/m): -0.0098 in air the sun stirs up |
| `temperature_log_k` | a change of temperature fastest near the ground, in K times ln(1 + z / `roughness_m`): negative on a sunny day, positive on a clear night |
| `roughness_m` | the ground's roughness length for the log profiles (m, 0.05 by default; 0.01 to 0.1 over grass, up to 1 among buildings and trees) |
| `table` | a measured profile in place of the formulas: rows of [height above the ground (m), temperature (°C), wind speed (m/s)] by height, linear between them and as at the top above them; the wind blows along `wind.direction_deg` |

The scene's `air.temperature_c` is the temperature at the ground (a table gives its
own). The wind and the temperature depend on the height only, the same over the whole
scene: this suits open ground better than streets, where buildings shelter the wind and
channel it. See [`examples/open_air_stage/scene.yaml`](https://github.com/naelolaiz/acoustic-spaces/blob/main/examples/open_air_stage/scene.yaml):
a stage heard 100 and 300 m away across a park, downwind and upwind.

### How the simulation bends sound

Along each direction from a source, sound travels at the effective speed
c(z) = c_T(z) + u(z) cos a: the speed of sound in still air at that height, plus the
wind's speed times the cosine of the angle a between the wind and the way the sound
goes (faster downwind, slower upwind, as in still air across the wind). This speed is taken as
linear in height between 160 heights from 1 cm to 1 km, closest together near the
ground, where the profiles change fastest, and constant above. Within each layer a ray
is an arc of a circle, so its travel time and its length are exact.

- **The direct sound and its reflections off the open ground** are the eigenrays: the
  curved paths from the source to the listener, straight there or off the ground up to
  ten times. Downwind there may be several where still air has one; in the shadow there
  are none. Each comes in with its own travel time and angle, at a level set by how far
  its neighbouring rays spread apart (1/R² for a straight path), fitted over the rays
  that turn back in the air a few layers above and below it, as the layers' corners
  would otherwise make it jump. Where neighbouring rays cross (a caustic) the level is
  capped at 10 dB over a straight path's, and each caustic delays the phase by a
  quarter period. Against an independent ray tracer on the smooth profiles, the levels
  of these paths are within 1 dB, 1.5 dB near a caustic.
- **In the bands up to 500 Hz** rays fail over the ground, as the profiles change most
  within a wavelength of it, where the direct sound and its reflection meet. There the
  sound over the open ground comes from a parabolic equation (the Green's function
  parabolic equation of Gilbert and Di) marched out from the source in the vertical
  plane through the listener, over the grounds along the way and their impedances. In
  the shadow, and from halfway to its start, where rays come out too loud, it gives the
  bands up to 2 kHz, and above them the shadow deepens with frequency as it does over
  the octave below. It applies from 1.5 times the source's and the listener's heights
  added away along the ground, where the sound comes in at 34 degrees from the
  horizontal or flatter; nearer, the atmosphere has had little room to bend it, and the
  rays give it. In still air the parabolic equation is within 0.25 dB of the exact
  solution over grass and harder grounds (see
  [Validation](validation.md#ground-reflection-against-the-exact-solution)).
- **Off façades and barriers** the paths bend the same way, in the vertical plane the
  reflections fold them into, with the wind's mean component over their legs. A path
  that would pass over the wall, or through something on the way, is left out.
- **Over an edge** (a roof, the top of a barrier) each leg keeps its straight course and
  takes the travel time of its heights, and past the start of the shadow its level
  falls as a creeping wave's does.
- **The direct sound's time** in the results is when the bent direct sound arrives:
  sooner downwind, later upwind (the time along the ground in the shadow).
- **The late rays** follow their arcs through the layers, in chords that stray at most
  2 cm from them, reflect where they meet a surface, and leave the scene once they are
  bent up out of it. The energy they rain onto listeners goes straight, in the time of
  the heights it crosses, and does not reach a listener in the shadow of where it
  leaves. These rays run on the CPU: `simulation.late.device: gpu` or `auto` falls back
  to it, and says so.
- **The wave solver** (`simulation.wave`) steps the sound field itself in its bands, with
  a speed of sound that changes from cell to cell: at each cell's height, the still
  air's plus the wind's component along the horizontal from the source, as above, so
  the rays, the parabolic equation and the wave solver bend sound alike. Its grid is
  fine enough for the slowest sound and steps in time for the fastest, a few percent
  more cells and steps than in still air in the `windy` preset. Over grass, 50 to 200
  m from the source, in the 63 and 125 Hz bands, it is within 0.7 dB of the parabolic
  equation, 0.2 dB on average (see [Validation](validation.md#the-wave-solver-in-wind-and-temperature-against-a-parabolic-equation)).
  It runs on the CPU: `simulation.wave.device: gpu` or `auto` falls back to it, and
  says so. The run's details give the slowest and the fastest speed of sound on the
  grid (`wave_speed_min_m_s`, `wave_speed_max_m_s`).
- **The animation's particles** and **the drawn paths** (`view --paths`, the Blender
  export) curve as the sound does.

Bending takes time: the `open_air_stage` example runs in 3 s instead of 0.2 s in still
air, a scene's late rays take about three times as long, and coverage maps about six
times, the parabolic equation running for each point; `simulation.performance.threads`
shares the work out over the cores. An atmosphere in which sound goes straight (`neutral`, no wind
and no temperature gradient) changes nothing: the results are those of still air, bit
for bit.

**Without turbulence, shadows come out deeper than measured.** The eddies of the wind
and of the temperature scatter sound into a shadow and keep it to some 20 to 30 dB in
practice; without them a shadow far from its start can be 60 dB deep at high
frequencies. Give the atmosphere its `turbulence` (see [Turbulence](#turbulence)).

Against measurements of a wind turbine up to 845 m away in light winds, the bands from
80 Hz to 2 kHz are within 1.7 to 2.7 dB on average, about as close as the published
parabolic equation model (see [Validation](validation.md#outdoor-propagation-against-measurements-hannover)).

`acoustic-spaces inspect` describes the atmosphere and gives the speed of sound across
the wind, downwind and upwind at a few heights up to the sky. From Python,
`acoustic_spaces.viz.plot_atmosphere(scene.atmosphere, scene.air.temperature_c,
"profile.png")` draws the temperature, the wind and the speed of sound over the ground.

## Turbulence

The wind and the temperature are not steady: eddies of every size, from the gusts of
the wind to the warm plumes the sun raises off the ground, drift past, and the speed of
sound flickers with them. Outdoors that matters in three ways:

- The paths of one sound that go different ways stop adding in step: the **ground dip**,
  where the direct sound and its reflection off the ground cancel, fills in at long
  distances and high frequencies.
- Sound is **scattered into shadows**: a shadow of the wind or of the sun's warmth
  stays some 15 to 30 dB deep instead of 60.
- The sound of a far source **flickers**, louder and softer by several decibels from
  one second to the next, as the eddies drift past.

```yaml
atmosphere:
  preset: windy
  turbulence: moderate          # weak, moderate or strong; or the values themselves:
  # turbulence: {temperature_structure: 0.008, wind_structure: 0.12, outer_scale_m: 10}
```

`turbulence` goes in the `atmosphere`, also in still air (`atmosphere: {turbulence:
strong}`). Its fields:

| field | what it sets |
|---|---|
| `preset` | a strength to start from (below); fields given with it change it |
| `temperature_structure` | the temperature's structure parameter C_T² (K² m^-2/3): about 0.001 to 1 near the ground, the most in sunshine |
| `wind_structure` | the wind's structure parameter C_v² (m^4/3 s^-2): about 0.01 to 1 near the ground, the more the stronger the wind |
| `outer_scale_m` | the size of the largest eddies (m, 10 by default): some metres near the ground |

| preset | C_T² | C_v² | weather |
|---|---:|---:|---|
| `weak` | 0.001 | 0.03 | overcast, a light wind |
| `moderate` | 0.008 | 0.12 | a breeze and some sun |
| `strong` | 0.08 | 0.5 | a strong wind or a hot sun |

These are indicative: the eddies are strongest near the ground and weaken with height,
and they are taken here as the same at every height, at the strength near the
ground. Without `turbulence`, or with both structure parameters 0, the results are
those without eddies, bit for bit.

### How the simulation takes eddies into account

The eddies make the index of refraction fluctuate, by -T' / (2 T0) for the
temperature's and -v' / c0 for the wind's, both with a von Kármán spectrum of outer
scale `outer_scale_m`, taken together as one of strength Cn² = C_T² / (4 T0²) + (11/6)
C_v² / c0² (Ostashev and Wilson 2015).

- **Each two early paths of a sound** add in pressure times their coherence factor
  exp(-D / 2): D grows with the square of the frequency, with the distance and with how
  far apart the two paths go, from the Markov approximation along both (Clifford and
  Lataitis 1983) at 24 points each. Near the ground the eddies that part the direct
  sound and its reflection span few wavelengths, and a parabolic equation marched
  through random eddies fills the dip far less than that: two paths that differ only by
  reflections off the ground take D times 1 / (1 + (32 / k rho)^1.7), rho their largest
  separation, fitted to it. With it the level of the direct sound and its reflection is
  within about 1 dB of the parabolic equation's on average over eight grounds and
  heights (see [Validation](validation.md#turbulence-against-a-parabolic-equation-through-random-eddies)).
  The echogram, the parameters and the coverage maps take this average.
- **In the parabolic equation's bands** (in wind or temperature gradients, see
  [How the simulation bends sound](#how-the-simulation-bends-sound)) the eddies blur its
  sound towards the rays' added in energy, by the rays' coherence factors.
- **In a shadow** the eddies add the sound they scatter into it: once, from the air
  that the source and the listener both see over the shadow's edge (the Born
  approximation, which the parabolic equation through random eddies bears out at 250
  Hz), and at higher frequencies by shaking the refraction near the ground that makes
  the shadow, a share of the free field's energy that grows as the square of the eddies'
  speed of sound against the fall of the speed of sound over the heights. That share is
  fitted to the parabolic equation on a sunny day and upwind in the `windy` preset, the
  source 2 m and the listener 1.5 m up, and deep in those shadows the engine is within
  about 1 dB of it on average, 3.5 dB at most; the shadows of other heights and profiles
  follow the same formula, untested. A barrier's or a building's shadow takes nothing from the
  eddies.
- **An impulse response is one moment of the eddies.** Each path is delayed at random,
  band by band, as the eddies have it at that moment (the seed draws it), so the levels
  of a response scatter about the echogram's average, the more the stronger the eddies
  and the higher the band; averaged over many moments they give it. In the parabolic
  equation's bands a response carries the average.
- **Listening and walkthroughs flicker.** `acoustic_spaces.render.auralize`, the
  editor's **Listen** and `walkthrough` send the sound through a new moment of the
  eddies every sixteenth of the time they take to drift past (the outer scale over the
  wind's speed, at least 1 m/s; at most every 50 ms), each crossfaded into the next; the
  moments change as an Ornstein-Uhlenbeck process, so the delays drift rather than
  jump, and a walker keeps the same eddies from place to place. The `auralize` command
  and live playback (`play`, head tracking) go through the saved responses, one moment
  each.

## Vegetation

Trees, hedges and belts of trees are no surfaces: sound passes through them, and their
leaves, twigs and trunks take some of it on the way, absorbing it or scattering it on.
`geometry.vegetation` lists them:

```yaml
geometry:
  ground: {extent: [[-90, -30], [90, 90]], sky: 25}
  vegetation:
    - name: wood                   # a belt of trees: a footprint raised to a height
      footprint: [[-90, 15], [0, 15], [0, 45], [-90, 45]]
      height: 15
      foliage: dense               # ISO 9613-2's dense foliage, the default
      stems: {per_m2: 0.08, diameter: 0.3}
    - name: hedge                  # a hedge: a path of a width
      path: [[10, 50], [40, 50], [40, 60]]
      width: 1.5
      height: 2
    - name: oak                    # a lone tree's crown
      tree: [45, 52]
      crown_diameter: 10
      base: 3                      # where the crown starts
      height: 14
```

| field | what it sets |
|---|---|
| `name` | its name (`vegetation_1`, `vegetation_2`... by default) |
| `footprint` | a belt of trees or a wood: the (x, y) corners of the ground it covers, raised from `base` to `height` |
| `path`, `width` | a hedge: the (x, y) points along it and how wide it is (its sides mitred at the bends) |
| `tree`, `crown_diameter` | a lone tree: where it stands and how wide its crown is; the crown is a spheroid from `base` to `height` |
| `height` | its top over the ground |
| `base` | its bottom over the ground (0 by default); for a tree, where the crown starts, the trunk below left out |
| `foliage` | what the leaves and twigs take: a preset or values (below) |
| `stems` | the trunks and stems in it: `per_m2` per square metre of ground, `diameter` across (m) |

Each has one of `footprint`, `path` or `tree`, in the geometry's units over the ground.
A volume may run past the ground's edge (a wood the ground cuts off), and where two
overlap, what fills them adds up.

**Foliage** is a preset or an attenuation per metre:

| `foliage` | what it takes |
|---|---|
| `dense` (the default) | ISO 9613-2's dense foliage (Annex A): 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09 and 0.12 dB per metre from 63 Hz to 8 kHz, all absorbed |
| `none` | no foliage: bare stems, a wood in winter (then it needs `stems`) |
| `{attenuation: [...], albedo: 0.5, asymmetry: 0.3}` | dB per metre per octave or third-octave band; `albedo` is the share of it that the foliage scatters on instead of absorbing (0 by default), `asymmetry` the mean cosine of the angle it scatters through (0 every way alike, towards 1 onwards) |

ISO 9613-2's rates are what dense foliage takes from sound passing through it in all,
with what it scatters onwards counted in, so the preset absorbs all of it and adds no
scattered sound. The standard gives them for 20 to 200 m of foliage, a fixed 1 to 3 dB
for 10 to 20 m and none below; the engine takes the rates per metre at any depth, so a
belt under 20 m deep loses a little less than the standard says, and it measures the
depth along the straight path, where the standard takes an arc of 5 km radius.

**Stems** are upright rigid cylinders. They absorb nothing and scatter, per metre of
horizontal travel, their number times a cylinder's scattering width: the exact series
for sound at right angles to its axis (Morse and Ingard 1968), averaged over the band.
0.08 stems per m² 0.3 m across scatter about 0.003 dB per metre at 63 Hz and 0.2 dB at
8 kHz. Sound going up or down meets them along its horizontal share only, and they turn
it sideways, keeping its slope: in the low bands mostly back the way it came, in the
high bands mostly onwards.

### How the simulation takes vegetation into account

- **Early paths** (the direct sound, the reflections, the paths round edges) lose
  exp(-Σ μ l) of their energy, μ each medium's attenuation per metre (the stems' times
  the horizontal share of the leg) and l the length of each leg inside each volume:
  Beer-Lambert. In the parabolic equation's bands (in wind or temperature gradients)
  the sound over the ground loses what the direct path loses. What the plants scatter
  out of the early paths comes back in the late part.
- **The late rays** lose what the plants absorb along their way and scatter in them at
  random, as often as the media scatter. Where a ray scatters in foliage, the foliage
  sends its share straight to each listener the way its phase function has it
  (Henyey-Greenstein), as a surface's scattered sound reaches them, and the ray goes on
  in a direction drawn from it; where it scatters off stems, it turns sideways, keeping
  its slope, so the stems send a listener nothing straight: they send what the ray's
  next leg would carry through the listener's detector sphere, averaged over the turns
  (a quadrature over those that cross it). The rain from every reflection loses what
  the plants take on its way to the listener. In several bands at once, the rays
  scatter at the rate of the bands' energy-weighted mean and each band is weighted by
  what it meets, so each keeps its own expectation. Against a separate particle
  simulation through a slab of foliage or stems, the tracer agrees within its noise,
  a few tenths of a percent (see
  [Validation](validation.md#vegetation-against-a-particle-simulation)). Rays through
  vegetation are traced on the CPU: the GPU devices fall back to it.
- **Coverage maps** take the same: the early paths' loss and the late rays'.
- **The wave solver's bands** leave vegetation out: at their low frequencies foliage
  takes little (0.02 to 0.03 dB per metre in the `dense` preset).

Behind vegetation that scatters (stems, or foliage with an `albedo`), much of the sound
comes scattered, with the late rays, and their noise shows in the levels: trace more
of them (`simulation.late.rays`) for steadier results. The
[`tree_belt`](https://github.com/naelolaiz/acoustic-spaces/blob/main/examples/tree_belt/scene.yaml)
example traces 200000, and G behind its trees still varies from run to run, by 0.1 to
0.7 dB (one standard deviation) depending on the band.

`acoustic-spaces inspect` lists the vegetation with what sound loses per metre through
each, foliage and stems together. The editor's 3D view, `view` and the export to
Blender draw the volumes in see-through green, and coverage maps outline them.

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
  grid, as for any open model, and in wind and temperature takes a speed of sound per
  cell (see [How the simulation bends sound](#how-the-simulation-bends-sound)).
- **Air** absorbs per band as indoors (ISO 9613-1, from `air`), along every path.
- **Wind and temperature** bend the paths and the rays (see
  [Wind and temperature](#wind-and-temperature)), and their eddies blur the paths'
  interference, scatter sound into shadows and make it flicker (see
  [Turbulence](#turbulence)).
- **Vegetation** takes sound from the paths through it and scatters some of it into
  the late part (see [Vegetation](#vegetation)).
- **The early paths add in pressure in every band** (see
  [Ground impedance](#ground-impedance)): `simulation.early.coherent_max_band_hz` is
  8000 outdoors unless the scene sets it.

`acoustic-spaces inspect` says where the ground and the sky lie, describes the
atmosphere, and leaves out the statistical reverberation times, which need a diffuse
field.

## Results outdoors

Outdoors every response and every map point also gives its **level relative to free
field**: how much louder (or, negative, quieter) the source is there than the same
source as far away in free field, with the same air between them, per octave band and
A-weighted. The ground, the weather, buildings, barriers and vegetation make the
difference; with the sign turned it is the excess attenuation of outdoor acoustics. It
compares the energy the response holds with the free field's, so it needs no decay to
read, and neither does G outdoors: where a response does not decay as a room's does
(an open field, a lone barrier), G and the levels that take it are those of the energy
it holds instead of undefined; the other parameters stay undefined. `simulate` prints
it, `parameters.json` has `re_free_field_db` and `re_free_field_a_db`, maps have the
parameter `re_free_field` and the editor's Results panel the row **Re free field**.

The **insertion loss** of buildings, barriers or vegetation is what they take off the
level: the level without them less the level with them, per octave band and A-weighted
for the source's sound power. Name them in the scene,

```yaml
simulation:
  insertion_loss_of: [noise_barrier]   # buildings, barriers or vegetation, by name
```

or for one run on the command line (`acoustic-spaces simulate scene.yaml --without
noise_barrier`, and the same for `map`), and the scene is run as it is and again
without them, with the same settings and seed so that both runs trace the same rays; a
run takes twice as long. The results of the scene as it is then carry the insertion
loss at every listener (`insertion_loss_db` and `insertion_loss_a_db` in
`parameters.json`, printed by `simulate`) and at every map point (the map parameters
`insertion_loss` and `insertion_loss_a`), and the report says what was left out
(`removed`). A part without a name goes by its kind and its place in the list
(`building_1`, `barrier_2`, `vegetation_1`). A barrier that shields the listener has a
positive insertion loss, a building that sends sound back towards the listener a
negative one. In the editor, the **Simulation** entry of an outdoor scene has an
**Insertion loss** list with every building, barrier and vegetation to tick, and the
Results panel the row **Insertion loss**. From Python,
`acoustic_spaces.engine.insertion.simulate` and `coverage_map` take `without=[...]`, and
`Scene.without(names)` is the scene without them.

Both are checked against exact solutions: a barrier's insertion loss within 0.07 dB of
the Biot-Tolstoy-Medwin solution over an absorbing ground, the level over grass within
0.15 dB of the exact ground reflection outside the band-filtered dip (see
[Validation](validation.md#outdoor-results-against-exact-solutions)).

## In the editor

The **Geometry** entry of an outdoor scene has an **Outdoors** group: the ground's height,
the margin and the sky, where the ground ends, the buildings, barriers and vegetation
with their heights, which can be changed or removed (the vegetation's table also gives
each one's shape and what fills it), and the ground zones, which can be removed.
Footprints, paths, trees and the zones' outlines are set in the scene file. A material's page
has a **Ground impedance** group: tick it to make the material a ground (grass to
start with), then pick the ground, the model, a flow resistivity of your own, a layer's
thickness or the porosity rate. Unticked, the material keeps the absorption the
impedance gave it. The scene's own entry has a **Wind and temperature** group: the
weather (a preset, or custom), the wind's speed, direction, height and profile, the
temperature's gradient and its change near the ground, and the roughness; changing a
value makes the weather custom. Under them, **Eddies** picks the turbulence (none, a
preset or custom), with the temperature's and the wind's structure parameters and the
size of the largest eddies; they stay as the weather changes. A measured profile is set in the scene file, and
**Remove the measured profile** drops it; **Show the profile** plots the temperature,
the wind and the speed of sound over the ground, across the wind, downwind and upwind.
On the simulation page, **In pressure up to** goes up to 8000 Hz in an outdoor
scene. The 3D view and the export to Blender leave the sky out (the 3D view keeps its
edges, to show where the ground ends), and the status bar gives the ground's area
instead of a reverberation time. **Draw room** replaces an outdoor scene with a floor
plan after asking, and brings **In pressure up to** back to a room's default when it
was above 500 Hz. The **Simulation** entry of an outdoor scene with buildings,
barriers or vegetation has an **Insertion loss** list, where ticking them runs the
scene again without them (see [Results outdoors](#results-outdoors)), and the Results
panel has the rows **Re free field** and **Insertion loss**.

## What is not modelled yet

The wave solver takes the wind as the rays and the parabolic equation do, as a speed of
sound that the wind's component along the horizontal from the source changes, not as
air that flows (the linearised Euler equations): sound that a building sends back
towards the source travels at the speed of sound going away from it. It leaves out
vegetation and the eddies, and in wind and temperature runs on the CPU only.

Vegetation is volumes filled evenly: a lone tree's trunk is left out, the foliage's
attenuation is given, not derived from its leaves' size and density, and stems stand
upright at random. The rays through it are traced on the CPU only.

The wind, the temperature and the eddies are the same over the whole scene, and the
eddies are as strong at every height as near the ground. The
ground is flat and reacts locally: its impedance does not
change with the angle sound meets it at, which holds for grass, soil and snow, though
less for a thick layer of very soft snow.

Sound that reflects off the ground both before and after bending over an edge (two
reflections and a diffraction) is left out: `first_order` diffraction takes one
reflection, before or after the edge. Over a rigid ground that path carries a quarter
of the sound over a barrier, so the barrier's insertion loss comes out 0.3 to 0.8 dB too
high from 1 kHz up and a few decibels off in the bands below, where the paths
interfere; over grass and other soft grounds, whose reflections near grazing are weak,
the gap is smaller (see
[Validation](validation.md#outdoor-results-against-exact-solutions)).
