# Changelog

## Unreleased

**Outdoor scenes**

- `geometry.ground` makes a scene outdoors: a flat ground under an open sky, which lets
  all sound out, with `buildings` (footprints raised to a height, open underneath),
  `barriers` (thin screens along a path) and a model file's objects standing on it. The
  ground reaches `margin` past them or over a set `extent`, and the sky lies `sky` over
  it or `margin` over the highest of them. Surfaces are named after each building
  (`<name>_roof`, `<name>_wall_1`...) and by kind (`walls`, `roofs`, `buildings`,
  `barriers`, `ground`, `sky`).
- Nothing reflects off the sky, no edge of the sky or on the ground diffracts and the
  ground's size scatters nothing, so the ground reads as endless; the inside corners of
  buildings do not diffract either.
- Sources and listeners inside a building, below the ground, above the sky or past the
  ground's edge are reported as problems.
- Outdoors, `ir_length_s` counts from the latest direct sound, so far listeners are not
  cut off; the statistical late method, which needs a diffuse field, stops with an
  error; coverage maps cover the ground and leave out points inside buildings.
- The editor's **Geometry** form has an **Outdoors** group (ground height, margin, sky,
  buildings' and barriers' heights); the 3D view, the plots and the Blender export leave
  the sky out; `inspect` and the status bar describe the ground instead of a
  reverberation time.
- New example: `street`, traffic in a street between two buildings, heard on the
  pavement, on a balcony and in a garden behind a noise barrier.
- Rooms are unchanged: indoor scenes give the same results, bit for bit.

**Ground impedance**

- A material's `impedance` makes it a porous ground: a Nord2000 ground class (`snow`,
  `forest_floor`, `loose_soil`, `grass`, `compacted_soil`, `gravel`, `asphalt`, `water`)
  or a flow resistivity, by Miki's model (the default), Delany and Bazley's or the
  two-parameter model, optionally as a layer on a hard backing. Its absorption is its
  random-incidence absorption, in third-octave bands.
- Reflections off it take the spherical wave's reflection factor (Weyl-van der Pol),
  with the phase and the ground wave near grazing incidence that make the ground dip:
  within 0.05 dB of the exact solution over grass and harder grounds, 0.2 dB over snow
  and 0.4 dB over a layer of snow 10 cm thick, outside dips more than 20 dB deep.
- Ground zones (`geometry.ground.zones`): parts of the ground with materials of their
  own, such as a road across a meadow. A reflection near where two grounds meet takes
  each one's factor, weighted by the share of its Fresnel zone each covers, band by band.
- Outdoors the early paths add in pressure in every band by default
  (`coherent_max_band_hz: 8000`; a room takes 500 at most), and the paths of each point
  of a line or surface source add in pressure, the points in energy.
- The wave solver fits a ground to its impedance itself, phase and all, with branches
  without mass: within 0.12 dB of the exact solution over grass from 40 to 315 Hz.
- The editor's material page has a **Ground impedance** group; the **Outdoors** group
  lists the ground zones; **In pressure up to** goes up to 8000 Hz outdoors. `inspect`
  shows each material's impedance and the zones; `--coherent-band` takes 1000 to 8000
  outdoors. `calibrate` leaves materials with an impedance alone.
- New example: `roadside`, traffic on a road between a meadow and a paved yard, with the
  ground dip over the grass. New validation: `validation/ground_reflection.py`.

**Wind and temperature**

- `atmosphere` (outdoor scenes): the wind and the temperature over the ground, which
  bend sound down (downwind, a clear night) or up into shadow zones (upwind, a sunny
  day). Presets `neutral`, `windy`, `sunny_day` and `clear_night`; the wind's speed,
  direction, height and log, linear or constant profile; a temperature gradient and a
  log term near the ground; the ground's roughness; or a measured table of heights,
  temperatures and wind speeds.
- The direct sound and the reflections off the open ground are the eigenrays of the
  effective speed of sound, curved paths with exact travel times, levels from the ray
  tubes' spreading (capped at caustics) and the phase lost at each caustic: several
  downwind, none in a shadow. Paths off façades bend in their unfolded plane; paths
  over edges fall into the shadow as a creeping wave.
- Below 500 Hz, and up to 2 kHz in and near a shadow, the sound over the open ground
  comes from a Green's function parabolic equation over the grounds' impedances, with
  Greene's starting field: within 0.25 dB of the exact solution in still air over grass
  and harder grounds. It runs on `simulation.performance.threads`.
- The late rays follow their arcs through the layers; the animation's particles,
  `view --paths` and the Blender export curve too. The direct sound's time in the
  results is the bent path's.
- The editor's scene page has a **Wind and temperature** group with **Show the
  profile**; `inspect` describes the atmosphere and the speed of sound at a few
  heights; `acoustic_spaces.viz.plot_atmosphere` draws it.
- New example: `open_air_stage`, a stage heard 100 and 300 m away across a park,
  downwind and upwind in a 5 m/s wind. New validation:
  `validation/outdoor_propagation.py`, a wind turbine heard 178 to 845 m away (the
  Hannover data, downloaded on first use): 1.7 to 2.7 dB per band on average in light
  winds, as close as the authors' own model.
- Scenes without an atmosphere, or with one in which sound goes straight, give the same
  results as before, bit for bit.

**Turbulence**

- `atmosphere.turbulence`: the eddies of the wind and of the temperature, by their
  structure parameters C_T² and C_v² and the size of the largest eddies, with a von
  Kármán spectrum; presets `weak`, `moderate` and `strong`. Also in still air.
- Each two early paths of a sound add in pressure with their coherence factor, from the
  Markov approximation along both, reduced for paths that differ only by reflections
  off the ground as a parabolic equation marched through random eddies has it: the
  ground dip fills in at long distances and high frequencies, within about 1 dB of that
  parabolic equation on average over eight grounds and heights. The echogram, the
  parameters and the coverage maps take this average; in the parabolic equation's bands
  the eddies blur its sound towards the rays' added in energy.
- Eddies scatter sound into the shadows of the wind and of the temperature: once from
  the air over the shadow's edge (the Born approximation), and at high frequencies by
  shaking the refraction near the ground: within about 1 dB of a parabolic equation
  through random eddies on average. 300 m upwind in the `windy` preset the bands from
  1 kHz up are about 19 dB below those downwind with moderate eddies, 60 without.
- An impulse response is one moment of the eddies (drawn with the seed), each path
  delayed band by band; `auralize`, the editor's **Listen** and `walkthrough` go through
  a new moment every sixteenth of the time the eddies take to drift past (at most 20
  times a second), crossfaded, so the sound flickers.
- The editor's **Wind and temperature** group has **Eddies** (none, a preset or custom)
  with the structure parameters and the eddies' size; `inspect` describes them.
- The `open_air_stage` example has moderate eddies. New validation:
  `validation/turbulence.py`, against a parabolic equation through random eddies;
  `validation/outdoor_propagation.py --turbulence` adds eddies to the Hannover cases.
- Scenes without turbulence give the same results as before, bit for bit.

**Vegetation**

- `geometry.vegetation` (outdoor scenes): belts of trees and woods (a footprint raised
  to a height), hedges (a path of a width, mitred at its bends) and lone trees (a
  spheroid crown), which sound passes through. Each holds foliage, ISO 9613-2's `dense`
  preset (the default, all absorbed) or an attenuation per band with an albedo and a
  mean cosine of scattering, and stems, upright rigid cylinders by their number per m²
  and their diameter, which scatter by the exact series and only sound's horizontal
  share. Overlapping volumes add up, and a volume may run past the ground's edge.
- The early paths lose what the plants take along each leg (Beer-Lambert), the
  parabolic equation's sound what its direct path loses. The late rays absorb along
  their way and scatter in the plants at random: foliage by Henyey and Greenstein's
  phase function, with its share sent straight to each listener, stems sideways,
  keeping the slope, with what the next leg carries through each listener's sphere
  averaged over the turns; in several bands at once each band keeps its own
  expectation. The rain from every reflection loses what the plants take on its way.
  Coverage maps take both. The wave solver's bands leave vegetation out.
- The editor's **Outdoors** group lists the vegetation (its shape, what fills it, its
  heights, and **Remove**); the 3D view, `view` and the Blender export draw it in
  see-through green, coverage maps outline it, and `inspect` gives what sound loses per
  metre through each.
- New example: `tree_belt`, traffic on a road heard behind a belt of trees and across an
  open meadow. New validation: `validation/vegetation.py`, the late rays against a
  separate particle simulation through a slab of foliage or stems: within a few tenths
  of a percent.
- Scenes without vegetation give the same results as before, bit for bit.

**The wave solver in wind and temperature**

- Outdoors, in an atmosphere whose wind or temperature changes with height, the wave
  solver's speed of sound changes from cell to cell: the still air's at each cell's
  height, plus the wind's component along the horizontal from the source, the effective
  speed of sound the rays and the parabolic equation take. The grid is fine enough for
  the slowest sound and steps in time for the fastest; each cell weighs its neighbours
  by the square of its speed over the grid's, its walls by that speed (their admittance
  is relative to the local rho c) and the source by its square, and so does the matched
  layer. The flow itself (the linearised Euler equations) stays out.
- The run's details give the slowest and the fastest speed of sound on the grid
  (`wave_speed_min_m_s`, `wave_speed_max_m_s`).
- New validation: `validation/wave_outdoor.py`, the 63 and 125 Hz bands over grass, 50
  to 200 m from the source, in still air, downwind and upwind, on a sunny day and on a
  clear night, against the parabolic equation: within 0.7 dB at 63 Hz and 0.4 dB at
  125 Hz, 0.2 dB on average.
- Indoors, and outdoors in still air, the wave solver gives the same results as before,
  bit for bit.

**Outdoor results**

- Outdoors, every response and map point gives its level relative to free field, per
  octave band and A-weighted: how much louder the source is there than the same source
  as far away in free field, with the same air; with the sign turned, the excess
  attenuation. `simulate` prints it, `parameters.json` holds it (`re_free_field_db`,
  `re_free_field_a_db`), maps have the parameter `re_free_field` and the editor's
  Results panel the row **Re free field**.
- `simulation.insertion_loss_of` names buildings, barriers or vegetation (`--without`
  on the command line, for `simulate` and `map`; an **Insertion loss** list on the
  editor's Simulation page): the scene is run again without them, with the same seed,
  and the results carry their insertion loss at every listener (`insertion_loss_db`,
  `insertion_loss_a_db`) and map point (`insertion_loss`, `insertion_loss_a`), the
  level without them less the level with them; the report says what was left out
  (`removed`). `Scene.without(names)` is the scene without them, and
  `acoustic_spaces.engine.insertion` runs both.
- Outdoors, G of a response that holds no decay to read (an open field, a lone
  barrier) is that of the energy it holds instead of undefined, and so are the levels
  and the STI that take it; the other parameters stay undefined. Indoors nothing
  changes.
- New validation: `validation/insertion_loss.py`, a barrier's insertion loss against
  the exact solution for a half-plane (within 0.07 dB over an absorbing ground) and the
  level over grass against the exact solution for a ground (within 0.15 dB, up to 1.1
  dB in the ground dip below 1 kHz, which the band filters fill), and a barrier over a
  rigid ground against the four paths over it (below).
- Scenes without `insertion_loss_of` give the same results as before, bit for bit,
  except that G outdoors is now defined where it was not.

**Reflections on both sides of an edge**

- `simulation.early.diffraction_reflections: both_sides` (`--diffraction-reflections`
  for `simulate` and `map`; **Reflections over an edge** on the editor's Simulation
  page) lets a diffracted path reflect both before and after the edge: off the ground
  on the way to a barrier and again behind it, the fourth of the four paths over a
  barrier on the ground, which carries a quarter of the sound over it on a hard ground.
  Its reflections take the materials' coefficients and phases as any other path's, and
  it counts as a diffracted path of order 2 (its two reflections) where paths are
  drawn, exported and reported.
- Validation (`validation/insertion_loss.py`): a barrier on a rigid ground against the
  four paths by the exact solution, the exact solution read through the engine's own
  band filters below 1 kHz: within 0.15 dB at every listener and band with
  `both_sides`; with `one_side` the insertion loss is 0.3 to 0.8 dB too high from 1 kHz
  up and up to 3 dB off below.
- `one_side` stays the default, so existing scenes give the same results, bit for bit.

**Foliage from the leaves' size and density**

- `foliage: {leaf_area_per_m3: F, leaf_width: a}` on a vegetation volume gives its
  leaves instead of a preset or measured rates: their area (one side's) per cubic
  metre and their width, the diameter of a disc of a leaf's area. The engine works out
  what they take in each band from thin rigid discs turned every way alike
  (`engine.vegetation.disc_scattering`, `leaves_medium`): a dipole, from the air a
  small disc holds back, 16 (kr)⁴ / (81 π²) of its area; twice the mean projected area
  of a large one, half the forward shadow and half reflected every way alike; the two
  bridged smoothly in between. They absorb in the viscous and thermal boundary layers
  on their faces (`leaf_absorption`, Morse and Ingard's rigid wall), a few per mille of
  their area. The scattered share and its mean cosine per band replace `albedo` and
  `asymmetry`. The editor's vegetation list and `inspect` show such leaves; the late
  rays, the early paths and coverage maps take them as any other foliage.
- New validation: `validation/leaves.py`, the disc's cross-section against its closed
  forms at both ends, the absorption against the wall's in closed form, and a wood of
  leaves 50 m deep compared with ISO 9613-2's dense-foliage table: leaves 5 cm wide at
  0.1 m² per m³ take about what the table gives at 4 kHz (3 to 3.5 dB against 4.5),
  more at 8 kHz and next to nothing below 1 kHz, where the table's 1 to 3 dB are not
  what leaves scatter. `validation/vegetation.py` runs the late rays through such
  leaves against its particle simulation too.
- The `dense` preset, `none` and measured `attenuation` give the same results as
  before, bit for bit.

**Terrain and buildings from a map**

- `ground.terrain` makes the ground uneven: its heights on a square grid, from an Esri
  ASCII grid or a table of x y z points (`file`; `gdal_translate -of AAIGrid` turns a
  GeoTIFF into the first) or given in the scene (`heights`, `origin`, `spacing`),
  bilinear between the nodes; `ground.map_origin` is subtracted from a map's
  coordinates, and `ground.height` added to the heights. The scene's ground is the grid
  sampled over its extent (the whole grid unless an extent is set) and cut into
  triangles, with the sky walls rising from its edge; buildings, barriers and
  vegetation stand on it however it runs, and ground zones take its triangles. Files in
  degrees of longitude and latitude are refused. `inspect` describes the terrain and
  gives each source's and listener's height over the ground.
- `geometry.buildings_file` adds the buildings of a GeoJSON file: `Polygon` and
  `MultiPolygon` footprints raised to the height a property gives (`height`,
  `default_height`), named by a property (`name`), some left out (`without`); holes are
  built over and reported. The editor lists them after the scene's own buildings and
  removes one into `without`.
- Over a terrain the early sound reflects once off the mean plane of the ground's
  profile between the source and the listener (CNOSSOS-EU's and Nord2000's ground
  reflection, with the ground's reflection factor under the reflection point), not off
  the terrain's facets (`engine.terrain`), and bends over the crests the line of sight
  cuts through (`engine.crests`): the string pulled taut over the profile, one wedge of
  the uniform theory of diffraction per crest with the ground's reflections off the
  profile's segments on both sides in its terms, Fresnel-zone weighted as in Nord2000
  (a term of the wedge's coefficient near the crest, a path of its own far from it),
  the equivalent edge of a flat or rounded top with ISO 9613-2's thick-screen
  factor over the ground within an eighth of a wavelength of the string (nothing for a
  ridge), and in sight the nearest crest's wave with the transition at its boundaries
  and the reflection off the slope before it, so that the level is continuous across
  the edge of a hill's shadow and ripples in the lit zone as the exact field does. The
  late rays bounce off the terrain's triangles. Edges buried in a terrain do not
  diffract.
- New validation: `validation/terrain.py`, a flat terrain against the flat ground (the
  same paths to rounding), a plane slope against the image source in the tilted plane,
  a ridge against the exact solution for the wedge (within 0.2 dB in every band), a
  smooth hill against the thin screen's attenuation (Maekawa's formula over a soft
  ground, ISO 9613-2's over a rigid one, where the flanks' reflections add)
  and the level's steps as a listener climbs out of its shadow, and the cost of a
  terrain 2 km across (see [Validation](docs/validation.md#terrain-against-closed-forms-and-the-exact-wedge)).
- New example: `hillside`, a works yard at the foot of a hill heard in the hill's
  shadow, on its slope and on its crest, over a terrain from a grid of heights with two
  houses from a map.
- Scenes without a terrain or a buildings file give the same results as before, bit
  for bit.

**Outdoor scenes in the editor**

- **File > New outdoor scene** starts a flat meadow 200 m across under an open sky,
  with a source and a listener 50 m apart, to build on.
- The **Geometry** form's **Outdoors** group draws new parts in the plan view: **Add
  building**, **Add zone** and a belt of trees as outlines closed on their first corner,
  **Add barrier** and a hedge as paths ended with Enter or a double-click, and a lone
  tree with one click; each takes the usual sizes (a building 10 m high, a barrier 3 m,
  a belt 12 m, a hedge 2 m high and 1.5 m wide, a tree's crown 8 m across from 2.5 to
  10 m up, dense foliage) to change afterwards, and **Remove** takes it out again. The
  plan shows the vegetation in dashed green and the zones in their materials' colours.
- The ground's **Extent** is set in the form (or left to the margin), **Terrain from a
  file** and **Remove terrain** set and drop a terrain, and **Buildings from a map**
  adds a GeoJSON file's buildings after asking which of its properties hold their
  heights (or one height for all) and their names, with **Remove the map's buildings**
  to drop the file; `acoustic_spaces.geometry.terrain.footprint_properties` reads what
  a file offers.
- A saved scene refers to its terrain and map by paths relative to the scene file, like
  its model file.

**Validation against a fast field program**

- `validation/ffp.py`: a fast field program for sound over a flat ground through a
  layered atmosphere, written apart from the engine: the exact solution of the vertical
  problem through layers of constant sound speed under a Hankel transform, with its own
  Bessel functions, so it needs nothing but numpy. In still air it is within 0.001 dB of
  the exact solution for a locally reacting ground from 10 Hz to 1 kHz and 100 m to 10
  km, and its layers are converged.
- `validation/benchmark_1995.py`: the benchmark cases for outdoor sound propagation
  models (Attenborough and others, 1995): a source 5 m and a listener 1 m over a ground
  of flow resistivity 200 kPa s/m², 100 m to 10 km apart, in still air and in sound
  speed gradients of 0.1/s down and up, and the same gradients stopping 100 m up (the
  weather there is), at 10, 100 and 1000 Hz through the engine's parabolic equation and,
  as whole scenes, per octave band from 63 Hz to 1 kHz, against the program read as the
  engine reads its bands. The parabolic equation is within 0.1 dB of the program to 1 km
  and within 3 dB to 10 km with the gradient stopping 100 m up, as the weather's do, and
  parts from it beyond 1 to 2 km with the benchmark's gradient going on to a kilometre
  up; whole scenes are within 0.11 dB per band to 1 km in every case, in still air read
  through the band filters within 0.04 dB to 200 m. The comparison found the notch at
  the join of the parabolic equation's bands and the rays' fixed below; see
  [Validation](docs/validation.md#outdoor-sound-in-wind-against-a-fast-field-program).

**Outdoor scenes on the GPU**

- The GPU traces outdoor scenes as the CPU does, so `simulation.late.device: gpu`,
  `both` and `auto` take them now, where they fell back to the CPU: the rays bent by
  wind and temperature follow the same chords through the layers (the same turns and
  the same curve to the millimetre, ray for ray, worked in single precision without
  trigonometry, whose rounding on a GPU would move the chords' ends); the rays through
  vegetation meet the same foliage and stems, scatter and lose energy the same way, and
  carry the same per-band weights; the open sky lets them out (a surface into no room,
  which only the CPU took before); and the ray count is told from the chords it needs.
  Vegetation of more than 16 media stays on the CPU, which the run says. What reaches
  each listener in each band, and when, agrees with the CPU's within the noise of the
  ray count: `validation/outdoor_gpu.py` compares the two on the outdoor examples
  (see [Validation](docs/validation.md#outdoor-scenes-on-the-gpu)). The shader is
  compiled with only what a scene has of the outdoors, so rooms compile as fast as
  before. On Windows, Direct3D 12's shader compiler (FXC) compiles the GPU's shaders
  without its optimizer, which got the vegetation wrong (the driver optimizes them
  anyway, and they compile as fast); there a scene with both bent rays and vegetation
  stays on the CPU, which the run says, as FXC takes more than an hour over the tracer
  with both.
- The wave solver in wind and temperature steps its grid on the GPU too
  (`simulation.wave.device: gpu`, `both` and `auto`), with the speed of sound per cell,
  in its walls, its source and its matched layer, agreeing with the CPU to rounding.

**Eddies weaker with height**

- `atmosphere.turbulence.height_m` says the height above the ground the structure
  parameters C_T² and C_v² hold at (where they were measured or estimated, usually 2 m):
  from it the eddies weaken with height as (z / `height_m`)^-`falloff`, taken no nearer
  the ground than their outer scale. `falloff` is 2/3 by default, as the surface
  layer's structure parameters fall in neutral and stable air (Monin-Obukhov
  similarity), up to 4/3 for the temperature's in strong sunshine. Without `height_m`
  the eddies are the same at every height, as before: results do not change.
- The coherence of each pair of early paths takes Cn² at the mean height of the two
  points along them, so paths aloft part slower; the sound scattered into a shadow takes
  Cn² at the height it scatters at, and the shaking of the refraction near the ground
  Cn² at the geometric mean of the source's and the listener's heights.
  `validation/turbulence.py --profile` marches the parabolic equation through eddies
  strengthening right down to the ground: the ground dip comes out within 1.1 dB on
  average over the soft grounds and 1.4 over the hard ones, the shadows within 1.3 to
  2.9 dB; the thin layer of stronger eddies at the ground does not tell there, which is
  why the engine stops at the outer scale (see
  [Validation](docs/validation.md#turbulence-against-a-parabolic-equation-through-random-eddies)).
- `inspect` names the height and the fall-off; the editor's Wind and temperature group
  has **Eddies' height** (0: the same at every height) and **Weaker with height**.

**Grounds of extended reaction**

- `impedance.reaction: extended` lets sound travel sideways inside a porous ground, so
  that each reflection meets the impedance of its own angle, Zc n / sqrt(n² - sin² θ)
  with n the ground's complex index (and the layer's with it), as Attenborough's models
  have it. It tells for soft snow, moss, forest floors and thin layers: a layer of snow
  10 cm deep on ice reflects up to 3.7 dB differently 20 m away, fresh snow 1.0 dB,
  grass 0.1 dB. The early paths take it at each reflection's angle, within 0.4 dB of the
  exact field, the fast field program's with the ground's reflection at every angle (1.5
  dB over the snow layer at 20 m, where the locally reacting impedance is off by up to
  6.4); the parabolic equation takes it at grazing incidence, within 0.4 dB (1.1 over
  the layer). The default, `local`, is as before: results do not change. Not with
  `two_parameter`, which has no index; the wave solver's ground reacts locally (see
  [Validation](docs/validation.md#ground-reflection-against-the-exact-solution)).
- `inspect` names the reaction; the editor's Ground impedance group has **Reaction**.

**Weather zones**

- `atmosphere.zones` draws parts of the scene with weather of their own, as polygons in
  the plan: the lee of buildings or a wood, where the wind drops; a lake, over which
  the air lies cooler under an inversion; a car park, warmer. A zone gives any of the
  atmosphere's fields (`preset`, `wind`, the temperature's gradient and log terms,
  `roughness_m`, `table`) and `temperature_c`, the temperature at the ground there;
  what it leaves out it takes from the scene's atmosphere. Outside the zones the
  scene's weather holds, the first zone wins where two overlap, and the eddies are the
  scene's everywhere. Without zones nothing changes: results are as before, bit for
  bit.
- The parabolic equation marches through the zones, changing its speed of sound where
  its path crosses a border, and the wave solver takes each cell's zone; the early
  paths bend through the mean profile along their way, each zone's weighted by the
  length of the way over it (the equivalent profile of Nord2000 and Harmonoise), and
  the late rays through the mean weather over the scene's extent, each zone's weighted
  by its area. Checked against a Crank-Nicholson parabolic equation marched through
  the zones (`validation/weather_zones.py`): the engine's parabolic equation is within
  0.15 dB on average and 1.2 dB at most to 800 m over shelter and gusts downwind and
  upwind, an inversion past a lake shore and cool water downwind; the mean profile
  within 0.3 dB where only the temperature changes on the way, and 1 to 4 dB, 6.6 at
  most, where the wind changes (see
  [Validation](docs/validation.md#weather-zones-against-a-crank-nicholson-parabolic-equation)).
- `inspect` names each zone, its corners and its weather. The editor's Wind and
  temperature group has **Weather of** (the whole scene, or a zone), **Add weather
  zone**, which draws one in the plan, **Remove weather zone** and a zone's **Ground
  temperature**; the plan draws the zones dotted in sky blue, with their names.

**Levels over a long term**

- `long_term` gives weather classes and their shares of the time, and the scene is
  run in each: noise maps and assessments give the level over a year, in which each
  weather counts for the share of the time it holds (ISO 1996-2, CNOSSOS-EU,
  Nord2000). A class takes any of a weather zone's fields over the scene's weather
  and may have eddies of its own; its `share` is one number or one per period of the
  day. The periods are L_den's by default (day 12 hours, evening 4 with 5 dB, night 8
  with 10 dB); `long_term.periods` gives others. Every run shares the seed, and a
  class whose weather is the scene's takes the scene's run. Without `long_term`
  nothing changes: results are as before, bit for bit.
- At every listener and map point, per octave band and A-weighted: the level in each
  class, in each period (the energy mean of the classes by their shares), the
  long-term level (the energy mean of the periods by their hours), the rating level
  (L_den, or L_R with other periods), and the long-term level relative to free field
  and insertion loss. `simulate` prints them, `parameters.json` has them in a
  `long_term` entry per response, maps gain `long_term` and `rating_level`, and
  `inspect` lists the classes and periods. Checked by `validation/long_term.py`: the
  combination agrees with the formulas worked from each class's run to rounding, and
  the correction from the classes favourable to propagation to the long-term level
  follows ISO 9613-2's C_met within 0.8 dB, with a C_0 of 3.4 dB (see
  [Validation](docs/validation.md#levels-over-a-long-term)).
- The editor's **Weather of** lists weather classes as well as zones, with **Add
  weather class**, **Remove weather class** and each class's **Share of the time** per
  period (the other classes make room in proportion); the results panel gains a
  **Long-term level** row and the run details each period's and class's levels and
  the rating level; maps offer both new parameters.

**Moving sources and Doppler**

- A walk with `doppler: true` (a car, a train, an aircraft, or a listener passing
  quickly) follows each early path from place to place and plays it through a delay
  that changes as the walker goes, read through a windowed sinc, so each path's pitch
  shifts as its length changes; its filters change from place to place as before, in
  turbulence the eddies drift past and binaural takes the ears' filters. The late part
  is crossfaded as before. A walk without it renders as before, bit for bit.
- Checked by `validation/doppler.py` against the exact field of a source passing over
  a rigid ground: within 45 dB at 30 m/s and 38 dB at 80 m/s, and the level within
  0.42 dB at every moment, where crossfading misses it by up to 11 dB; over grass the
  sound keeps the energy of the places' responses within 0.07 dB in every band (see
  [Validation](docs/validation.md#moving-sources-against-the-exact-field)).
- Pass-by levels: `acoustic_spaces.walkthrough.pass_by` gives the level at the
  listener over time (A-weighted and per octave band, the source standing at each
  place), its maximum, the sound exposure level and L_Aeq; the walkthrough of a walk
  with Doppler prints them, `walkthrough --levels CSV` saves the levels of any walk,
  and `inspect` says which walks have Doppler. Faster than sound or with the wave
  solver's bands, a walk with Doppler stops with an error.

**Moving air in the wave solver**

- `simulation.wave.wind: moving` (`simulate --wave-wind moving`, **Wind** on the
  editor's Simulation page) makes the wave solver's air move with the wind instead of
  changing its speed of sound: each cell keeps the still air's speed of sound, and
  before each step the pressure is carried along with the wind (fourth-order
  differences, a third-order Runge-Kutta method). Sound goes at c + v downwind and
  c - v upwind whichever way it turned, so a façade's echo against the wind comes when
  it should. It costs three to four times as much, on the CPU (asked for the GPU, the
  run says so and stays on the CPU); the run's details say `wave_moving_air`. Left
  out (`effective`, the default), and in still air, nothing changes.
- Checked by `validation/wave_wind.py` against the exact field in a uniform wind of 15
  m/s: within 0.05 dB and 0.1 ms in every direction, where the effective speed of
  sound is up to 0.9 dB off, and a façade's echo within 0.02 ms of ray theory's,
  where the effective speed of sound brings it 10.6 ms early; in the `windy` preset
  over grass within 0.34 dB of the parabolic equation, 0.13 dB on average, where the
  effective speed of sound is within 0.69 dB (see
  [Validation](docs/validation.md#moving-air-in-the-wave-solver)).

**Whole cities**

- Edge diffraction skips the pairs of an edge and a plane that no path reflecting
  between them can use, before it traces any ray: the edges behind the plane, and those
  whose shadow, cast on the plane from the source's or the listener's image, misses the
  plane's faces. For the listener's images it also skips the paths that would touch the
  edge's line beyond its ends or arrive too late. The exact tests run on the rest as
  before, in native code on `simulation.performance.threads` threads, so the results
  are the same bit for bit. A city of 900 buildings simulates in 2.3 s on four cores
  instead of 81 s and 7.1 GB, 1,600 buildings, which ran out of 14 GB, in 4.4 s, and
  3,600 in 22 s. The work still grows with the square of the buildings, as about 3 %
  of the pairs can carry a path as far as geometry alone tells.
- `validation/city_speed.py` times cities of growing size and checks the results
  against testing every pair (see
  [Validation](docs/validation.md#edge-diffraction-in-whole-cities)).

**Wide-angle parabolic equation**

- `simulation.early.parabolic_equation: wide_angle` (`simulate` and `map`
  `--parabolic-equation wide_angle`, **Parabolic equation** on the editor's Simulation
  page, for an outdoor scene in wind or temperature gradients) gives the bands the
  parabolic equation gives outdoors (up to 500 Hz, and in a shadow up to 2 kHz) from a
  wide-angle parabolic equation, `acoustic_spaces.engine.wide_angle`: the split-step
  Padé method of Collins' RAM, the one-way operator as a rational function of ten
  terms in steps of three wavelengths, fourth-order differences in height with the
  ground's impedance as the boundary, and the operator's own starting field, its
  tridiagonal solves in native code. It carries sound that a gradient reaching hundreds
  of metres up bends back at up to 45 degrees, through air whose speed of sound is far
  from the ground's, where the narrow-angle equation strays over a few kilometres. It
  takes three to four times the work; left out (`narrow_angle`, the default), nothing
  changes.
- Checked by `validation/benchmark_1995.py` against the fast field program: within
  0.11 dB at the benchmark's pure tones from 10 Hz to 1 kHz and 100 m to 10 km wherever
  the sound is within 40 dB of free field, where the narrow-angle equation strays by 3
  to 20 dB beyond 1 to 2 km; in the weather presets within 0.04 dB on average at the
  bands' centres and 0.02 dB over the bands (the narrow-angle one 0.11 and 0.08 dB);
  whole scenes of the benchmark's cases 2 and 3 within 0.03 dB per band to 2 km, where
  the narrow-angle equation is 0.5 to 0.8 dB off at 2 km downwind. Through weather
  zones `validation/weather_zones.py` finds it within 0.09 dB of the Crank-Nicholson
  reference on average (see
  [Validation](docs/validation.md#the-wide-angle-parabolic-equation)).

**Coherent arrays element by element**

- `near_field: true` on a source's array (`directivity: {file: line_array.yaml,
  near_field: true}`) adds up its elements' sound one by one wherever an early path
  leaves it, each from where it is, with its own distance, direction, feed and pattern,
  instead of sending out the array's far-field pattern from one point. Listeners nearer
  than about the array's length squared over the wavelength hear its sound as it is
  there: on axis a 3.75 m column loses about 3 dB per doubling of the distance up to
  41 m at 1 kHz, as a line does, and 6 dB beyond. The direct sound, the reflections,
  the ground's reflection and refracted rays outdoors, and the sound that reaches an
  edge all take it; the late reverberation keeps the pattern. The room no longer warns
  about such an array's near listeners, and the warning for the others names the
  option. In the editor: "Add up the array's elements one by one" under the source's
  directivity. Without it nothing changes. See
  [Validation](docs/validation.md#checks-against-theory-on-every-change).

**Groups, and sources left out**

- Sources and listeners take a `group` (nested with `/`, as `PA/Delays`) and
  `enabled: false`, which leaves them out of simulations, maps, walks' other sources
  and the scene's checks while keeping them in the scene; the others get the same
  responses as in a run of all. Named with `--source` or `--listener`, they are
  simulated all the same; `inspect` marks them "left out". See
  [Groups, and sources left out](docs/scenes.md#groups-and-sources-left-out).
- The editor's scene tree shows sources and listeners in their groups. A tick on each
  source, listener and group uses it in simulations or leaves it out, in one step of
  the history; an eye hides it in the 3D view and the plan or shows it again, without
  changing the scene. Sources and listeners left out are drawn faded. Their form has a
  **Group** field and **Use in simulations**.
- The `stadium_pa`, `concert_stage` and `stadium_concert` examples put their sources
  and listeners in groups: the PA's mains, subwoofers, fills and delays, the monitors,
  the stage, and the audience standing and seated.

**Examples**

- New examples of a concert's sound system, made of the same stage, arrays and boxes:
  - `stadium_pa`, a football stadium's sound system as tours hang it: two main line
    arrays of 12 boxes in a J, a block of eight cardioid subwoofers timed to arrive
    with them at the mix position, out-fills for the side stands, two in-fill arrays
    under the stage roof's front edge for the pitch's centre from 10 to 30 m, where the
    main arrays' horizontal coverage ends, front fills, two delay towers on the pitch
    and four delays under the roof's edge, each fill and delay arriving 6 to 20 ms
    after the main arrays at its seats (the in-fills with them), tuned at the mix
    position as a system
    engineer tunes it (the main arrays' low mids 1 to 2 dB down, their top octave 4 dB
    up for the air, the subwoofers 2 dB under them, the fills' low end rolled off). The
    stage's back is hung with drapes, as touring stages are, and the late sound reaches
    the listeners as secondary sources: in a bowl this size a detector sphere catches so
    few rays that a single one would be heard as an echo. Listeners stand on the pitch
    and on the stands. The crowds standing and seated scatter as a measured theatre
    audience does, rising from 0.3 at 125 Hz to 0.7 from 1 kHz (`scattering:
    theatre_audience`) rather than at one value; the rows stay out of the model, as
    surfaces smaller than the wavelength would only mirror sound falsely. `align.py`
    prints when and how loud each loudspeaker arrives at each listener, and which would
    be heard as an echo (none).
  - `concert_stage`, the stage alone on open ground with its monitors (wedges for the
    singer, a drum fill and side fills), the main arrays, the subwoofers and the front
    fills; the players and the mix position listen. Each player hears their own
    monitor loudest. `feedback.py` finds how much gain the singer's microphone allows
    before the loudspeakers make it ring, in each octave band at the mix position and
    at the singer's ears, the frequencies where it rings first, and renders the loop
    below and above that gain.
  - `stadium_concert`, the two together.

  Every array adds up its boxes one by one (`near_field: true`); `near_field.py` prints
  how far the arrays' far-field patterns would be off at each listener (within 3.5 dB
  up to 2 kHz, up to 10 dB at 4 and 8 kHz), skipping the bands a source's filter cuts.
  `make_model.py`, the same in the three, writes each one's model and arrays, of a
  generic box made up for the examples.
- The editor's File > **Open example** shows the examples in groups (rooms, halls and
  churches, sound systems, outdoors, models from CAD and BIM files, measurements and
  material libraries), and [Example scenes](docs/examples.md) lists them in the same
  groups; `acoustic_spaces.examples.example_groups()` gives them by group.

**Demos**

- Two more demos with their scripts, beside the orchestra in the concert hall:
  `church_demo.py`, in the `church` example, puts a choir of 19 real singers (VocalSet's
  dry takes, CC BY 4.0, brought onto the beat and towards the tune) and a sampled pipe
  organ (VCSL, CC0) in the church; `stadium_demo.py`, in `stadium_concert`, adds a band to
  the stage and plays a funk song through it, the sound system and the monitors, from the
  song's own stems ("Spicy Funk Cake", recorded for the DEGA, CC BY 4.0): the drum kit
  plays the drums, the singer the vocal, the guitarist's amplifier the guitar; its video
  turns each stop up to the loudness of the loudest, the stage, and says by how much, and
  its pictures show the crowd standing on the pitch and seated in the stands, the empty
  stand's seats, how many people stand and sit, and how much of the sound each takes up.
  `--song francium` or `rubidium` plays one of Software-Entwicklungskit's electronic songs
  (CC0) instead, from the stems its album is released with, all twelve listeners in the
  tour. Each downloads its recordings, writes its scene, mixes the piece at every
  listener with one gain for all and makes a video that tours them (with ffmpeg and
  Pillow).
- `orchestra_demo.py`, in `orchestra`, tours the concert hall's four seats through the
  Beethoven recording's three movements in one video: the seat changes at the
  quietest moment near each share of a movement, the movements follow each other with a
  pause, every seat's mix is shifted earlier by the time the sound takes to reach it,
  and all stops keep one gain, so the hall's differences are heard as they are; the
  cover says each stop's level against the loudest and carries a bar with the tour.
- `lakeside_demo.py`, in `lakeside_concert`, plays the same song from the lakeside stage
  through a sound system as a lawn concert hangs it (two line arrays of eight boxes and
  four cardioid subwoofers, aimed at the lawn and tuned at the audience) and mixes it in
  the audience, behind the stage, on a boat and in the village across the lake, in the
  evening's weather and in still air; its video plays the song at a bedroom
  window in the evening's weather, in still air and in the evening's again, each stop
  turned up to the audience's loudness with its real level in the caption, over a slice
  across the lake with the sound's rays bent by the weather.
- [Demos](docs/demos.md) gathers the four, with where each recording comes from, its
  licence and the credit a shared render needs; the orchestra's section moved there from
  [spatial audio and listening](docs/spatial-audio.md). The README and that page link the
  [playlist](https://www.youtube.com/playlist?list=PLWz5JDREfod8) of the videos.
- Every demo script takes `--hrtf me.sofa`, a SOFA file of your own or measured HRTFs:
  the `simulate` command it prints carries it, and `render` mixes from the responses
  made through that file, refusing ones made through the spherical head. A script asked
  for feeds or mixes before its scene exists says to run its scene step first.

**Editor**

- The wind is drawn outdoors (**View > Show wind**): sky blue arrows over the ground in
  the 3D view, in `view` and in the plan, the way it blows and as long as it is strong,
  at the height the scene gives its speed at, clear of the sources and listeners; each
  weather zone with its name and wind (and its ground temperature when it has its own),
  in the 3D view as see-through walls; and the scene's wind in a corner. **View > Show
  wind profile** stands a mast in the 3D view in each weather where the wind blows, with
  arrows up it and a curve: how the wind grows with height over the ground. The scene
  tree's new **Wind** entry has an eye for each and lists the weather zones, which
  show their weather in the scene's form when selected. Before, only the **Show the
  profile** plot showed the wind, and nothing showed a weather zone's.
- **File > Export particle animation to Blender** asks how much of the sound to animate,
  how slowly and how many particles, as `export --duration --slow-motion --count` do;
  outdoors it starts with sound enough to reach the farthest listener, slowed down to
  last about the default's 4 s. Before, it always took 100 ms, too little to leave the
  stage outdoors.
- New example: `lakeside_concert`, a concert on a lawn by a lake heard across the water
  from a boat and in a village 550 m away, with two weather zones: cool air and a
  stronger breeze over the lake, a weaker breeze among the village's houses. The
  editor draws its three weathers with **Show wind** and **Show wind profile**.
- While a run, a coverage map, a tuning or an export goes on, the status bar says what
  runs, its latest step and the time so far, with a **Cancel** button beside the bar;
  resting the mouse on the bar says the same at length. The bar was small and silent
  before, easy to miss on some desktops (KDE's, say).
- **Messages** shades every other message, and its ticks show or hide the info, the
  warnings and the errors, with how many there are of each. **Item** keeps only the
  messages about one source, listener or material, a filter only those holding some
  words, and **Clear** removes the messages on show.
- The markers of sources and listeners in the 3D view and `show` take their size from
  the scene: a fifth of the usual spacing between the points, from 1.2 % to 2.5 % of
  the model's size (it was 1.2 % always), so in most rooms they are twice as large.
  Where they would overlap they get smaller, down to a fifth, and an arrow that would
  run into a marker ahead of it stops short, in the plan too: on the stadium's stage
  the markers were 6 m wide and the arrows 19 m long, across the players and monitors.
  In `show`, sources and listeners left out are drawn faded too.

**Fixes**

- Binaural renders lost treble: at 8 kHz up to 13.6 dB from some directions (the
  spherical head's ear facing a source straight ahead), 2.7 dB on average over all
  directions, and more the higher the band. An HRTF between the measured directions
  mixes the responses round it aligned at their onsets, and an onset was the first
  sample within 20 dB of the peak: a response delayed by a fraction of a sample rings
  before its peak, more the nearer the fraction is to a half, so neighbouring onsets
  came out a sample or two apart and the mix cancelled the high frequencies. The onset
  is now the lag at which a response best matches its minimum-phase version (the same
  magnitude, starting at once), found to a fraction of a sample: the mix keeps the
  measured responses' spectrum within 1 dB. Mono, Ambisonics and loudspeaker outputs
  were not affected. Every binaural output changes; demo videos made before this sound
  duller than they should.
- A coverage map in the 3D view (and in `show`) hid the sources and listeners: they
  stand at listening height, half under its tiles. The map is now see-through.
- A reflection right where the sound came from, on a surface over about 100 m across,
  printed an overflow warning (an error under `-W error`). The results were right and
  stay the same.
- `auralize` and `walkthrough` (and `render.auralize` and `walkthrough.render_walk`)
  mix the sources at their sound power. The impulse responses carry each source's feed
  (gain, equalizer, delay) but are per unit of sound power, and the mix played every
  source as if it had the same `level_db`: in a sound system with fills 18 dB below
  its main loudspeakers, the fills came out 18 dB too loud. Each signal is now scaled
  by its source's `level_db` against the loudest one's in the mix; `--equal-power`
  (`equal_power=True`) keeps the old mix. A mix of sources of one power, or of one
  source, is unchanged. `parameters.json` lists each source's sound power
  (`sources`), which `auralize` reads; results written before it are mixed at one
  power, with a note.
- Levels, parameters and maps of open scenes take about half the time: the band filters
  that read the coherent bands were computed again for every band of every source and
  listener, and are now kept. The results are the same to the last bit.
- In wind and temperature gradients the impulse response had a notch where the
  parabolic equation's bands met the rays' (around 700 Hz, up to 3 dB deep; around
  2.8 kHz in the penumbra of a shadow), and the 500 Hz band of the level relative to
  free field came out 0.5 to 1.2 dB too low: the join between the two blended their
  spectra, which stand for different sound (the whole field over the ground on one
  path, each ray's own on the others), and the silent paths' higher bands counted
  against the band's reading. Each band of the response now keeps its own spectrum
  into the crossovers, and a path that carries nothing in a band is left out of that
  band's reading. The bands the parabolic equation gives now read back as computed,
  and the response follows the fast field program across the join within 0.3 dB. The
  same join in still air changes outdoor responses by less than 0.01 dB (one cell of
  the barrier tables by 0.2 dB, the ground dip at 250 Hz 200 m out) and rooms by
  under 0.0001 dB (the crossovers between the lowest bands' filters are drawn band by
  band, a difference of a hundred-thousandth of the peak).
- Saving an example elsewhere from the editor failed for the stadium ("directivity file
  not found: .../box.csv"): an array's YAML file names its element's pattern beside
  itself, and only the array file was copied. The files an array file names now go
  along with it (or, when one lies outside the example's folder, the scene keeps using
  the original array file), and a save that fails leaves no copies behind. Saving the
  hillside example also took its terrain and map of buildings along but kept pointing
  at the originals; it now uses the copies. `Scene.files()` lists every file a scene
  uses, the ones its array files name included. Saving also kept the window busy for
  seconds (Not Responding), adding up the arrays again from the copies; the copies
  hold the same bytes, so what was loaded stays: the stadium now saves in under a tenth
  of a second.
- The editor stopped answering while it exported a big run's results (Not Responding
  for over a minute with the stadium's 204 responses): the export ran in the window's
  own thread. It now runs in the background, with progress in the messages and
  **Cancel**, and `write_results` (and so `simulate --out`) renders and reads the
  responses on `simulation.performance.threads` (every core by default): the stadium's
  export takes 47 s instead of 81 s on 4 cores. The files and the report are the same
  to the last bit.
- Wheels left out the hillside example's terrain (`.asc`) and map of buildings
  (`.geojson`), so the installed example could not load. A test now checks that every
  file the examples use is installed.
- `second_order` diffraction printed a floating-point overflow warning where two
  edges meet at a corner and the shortest path over both runs through it. The path is
  still turned down, as it was, and the warning is gone.
- On a hardware GPU the wave solver could stop partway through a long run on a big
  grid: it sent up to 512 steps in one submission, and a GPU whose work runs longer
  than its watchdog allows (10 s on Linux's amdgpu, reached on an AMD Radeon 680M with
  two million cells) is reset, losing every step after it and failing the GPU runs
  that follow. The steps now go in submissions of about a second's work each, which
  step the grid exactly as one long submission did.
- The native code no longer fuses multiplies and adds that the source keeps apart, as
  Apple's compiler did by default on Apple silicon: results there now round as on the
  other platforms, last bits included.
- On Windows with Python 3.10 every GPU run printed four warnings that Vulkan lacked an
  extension (`Unable to find extension: VK_KHR_surface`...): wgpu 0.31, the last for
  Python 3.10, warns of each one on a computer without a Vulkan driver, where Direct3D
  12 does the work. They said nothing to a user and are left out now.

## 3.2.3

**Languages**

- Translations for Catalan and Italian for editor and command line text.
- German and Dutch translations with the same coverage: editor menus, forms, tips and
  messages, plots, examples' titles and descriptions, and the command line's language report.
  Select **Deutsch** or **Nederlands** under **View > Language** (from the next start),
  set `ACOUSTIC_SPACES_LANG=de` or `nl`, or follow the system language, including the
  Windows and macOS display language.

## 3.2.2

**Languages** (`acoustic_spaces.i18n`)

- The editor and the command line can show their text in another language, English or
  Spanish to start with. They follow the computer's language; **View > Language** in the
  editor picks one for the user (from the next start), and `ACOUSTIC_SPACES_LANG=es` or
  `en` sets it over both. `acoustic-spaces info` says which is in use and what chose it.
  Qt's own dialogs and buttons follow it too. Scene files, commands, options and the
  files the commands write stay in English.
- The editor is in Spanish: menus, panels, forms, tips, messages and the tuning window,
  the choices in its lists (methods, reception, spectra, shapes, directivities; the
  scene file keeps its values), the coverage map's parameters, the examples' titles and
  descriptions, and the plots' titles, axes and legends. Its numbers follow the
  computer's regional settings (0,85 in Spain, 0.85 in Mexico), in what it shows and in
  what it reads from its tables and fields.
- The Spanish text comes in steps: the command line and the library's messages (the
  room's checks and a run's progress, which the editor shows too) come next, and until
  then show in English.
- For contributors: text is marked with `_()` and collected into `locale/*.po` by
  `tools/catalog.py`, which CI runs to check that every catalog is complete; terms and
  style are in the new Translating page.

**Fixes**

- The editor no longer aborts with a wgpu panic (`egl.rs ... BadAccess`) when it looks
  for a GPU, as it does on opening the Simulation settings: with the `gpu` extra
  installed on Linux, wgpu also tried its OpenGL backend, whose set-up through EGL fails
  while the 3D view holds an OpenGL context, and that failure ends the process. wgpu
  now uses Vulkan, Metal or Direct3D 12 only, as the documentation says.

## 3.2.1

**Wheels for Intel Macs**

- Wheels for Intel Macs (macOS x86-64), on PyPI and the GitHub releases, so pip no
  longer builds the package from source there. They need macOS 10.15 or later. CI builds
  them on GitHub's Intel macOS runner and runs the tests and the editor's tests there
  too; pull requests now build the wheels for CPython 3.10 as well as 3.14.

**Fixes**

- Loudspeaker panning gives the same gains on every processor: an arrival on an edge
  between speakers could leave a speaker beyond that edge a gain of about 1e-17 from
  rounding (the rear height speakers of 5.x.4, for sound from straight behind, on Intel
  Macs). Such gains are now 0.

## 3.2.0

**On PyPI**

- `pip install acoustic-spaces` installs the package from PyPI, with the extras in
  brackets (`pip install "acoustic-spaces[viz,gui]"`). The GitHub release pages keep the
  same wheels. The description on PyPI is the README, with its links to the
  documentation and its picture pointing at GitHub.
- The repository is now github.com/naelolaiz/acoustic-spaces, the name of the package and
  of its command; the old address redirects. The documentation, the package's links and
  the editor's About window point at the new one.
- A message about a missing optional part says to `pip install "acoustic-spaces[gui]"`
  (with the extra it needs), where it listed the extra's packages for pip.

## 3.1.0

**Tuning the devices for a computer** (`acoustic-spaces tune`, `acoustic_spaces.engine.tuning`)

- `acoustic-spaces tune [scene]` (the concert_hall example by default) times the late
  stage on the CPU, the GPU and both, for responses of 0.5, 1, 2 and 4 s (`--lengths`)
  at the scene's ray count (`--rays`), and the wave solver on the CPU and the GPU. It
  prints the times and keeps the fastest for each in the user cache folder
  (`acoustic-spaces/devices.json`); `acoustic-spaces info` shows them. In the editor:
  Simulation > Tune CPU and GPU…, whose window saves on OK only (Cancel saves nothing).
- A per-computer default device: `simulation.late.device` and `simulation.wave.device`
  may be left out (the editor's **Run on**: *default*), and then follow this computer's
  default, `cpu` unless `tune --default auto` or the editor's tune window (a ticked
  check box) made it `auto`; `tune --default cpu` switches back. Scenes that name a
  device keep it. Scenes the editor saved before name `cpu`; the tune window offers to
  let the open one follow the default.
- The late stage's `auto` takes the choice for the run's response length. Without one,
  its trial traces the run's own length (it traced at most 0.25 s), and its choice is
  kept for that range of lengths. The 3.0.0 file `late-device.json` is no longer read.
- The wave solver's `auto` compares the timed costs for the run's step count, where it
  took any hardware GPU; without `tune` it still does.

**Warnings, errors and finished runs**

- The editor shows warnings in amber and errors in red, in **Messages**, in the scene's
  form and (the number of problems) in the status bar. On the command line, `error:`
  and `warning:` are red and yellow in a terminal (`NO_COLOR` turns that off).
- When a run, a coverage map or a tuning ends while another window is in front, the
  editor's taskbar entry flashes (the Dock icon bounces on macOS).

**Starting the editor**

- `acoustic-spaces` with no command opens the editor (`acoustic-spaces edit`), with the
  `gui` extra and a display; without them it prints the short usage and why, as
  before. `-h` still lists the commands.

**Tips in the editor**

- Resting the mouse on a label, field, button or column header of the editor shows what
  it means, with its unit: the Results table's rows say what each parameter measures,
  its band columns which frequencies each octave band holds, and **Band** which plots
  it chooses for. Menu entries and toolbar buttons say what they do (with their
  shortcut), and the scene tree's entries what they hold.

**Fixes**

- A binaural output through a SOFA file (`{format: binaural, hrtf: ...}`, or
  `simulate --hrtf`) gave no IACC_E and IACC_L in `parameters.json`, the printed table
  or the editor's Export; only the spherical head gave them. They now come from the
  first binaural output, whichever head it has. Such a response is written as
  `.binaural-<file name>.wav`, and `auralize --format binaural` finds it when it is the
  only binaural one, instead of asking for that name.
- Panning onto 5.x.4 and 7.x.4 layouts was lopsided. Their four height speakers, and
  the rear corners of 7.x.4, each lie on a plane, which was split along one diagonal, so
  20 to 28 % of all directions panned differently from their mirror image: a sound from
  straight above played from two diagonally opposite height speakers, and the
  reverberation of a symmetric room reached mirror-image speakers up to 2.8 dB apart.
  Four or more speakers on one plane now get an imaginary speaker at their middle that
  they play alike.
- Responses rendered at 44.1, 22.05 or 88.2 kHz, where 1 ms is not a whole number of
  samples, had random silent 1 ms gaps in their late part while it was still sparse (in
  a hall through its first 200 to 400 ms, losing up to about 15 % of the late energy):
  the noise's impulses were spread over slightly different slots from those its energy
  was set in. Rates where 1 ms is a whole number of samples, such as 48 and 96 kHz,
  were not affected and render as before.
- `calibrate()` took `lf`, `lfc` and `t30_x`, `t30_y` and `t30_z`, which its own error
  message offered, though the fit cannot work them out: it left them out, or with them
  alone returned the absorption it started from. It refuses them now, as
  `acoustic-spaces calibrate` did.
- Material libraries of CATT-Acoustic that give the 8 and 16 kHz values after a colon
  (`<a125 ... a4k : a8k a16k>`), as CATT writes them, could not be used at all, and
  `ABS1` materials (in fractions from 0 to 1) were left out. Both are read now.
- `acoustic-spaces walkthrough` ran on every core whatever the scene's
  `simulation.performance.threads` said, and took a negative `--threads`. Like
  `simulate`, it now takes the scene's setting unless `--threads` is given, and so does
  `simulate_walk`.
- The editor's Edit > Add material from a library… failed before showing its file
  dialog. It opens in the scene's folder now.
- A source with a `delay_ms` gave no stage support (ST early and ST late) at listeners
  within 1.5 m of it when the delay was more than about 1.5 ms at 1 m (4.4 ms
  anywhere): the delay counted as distance. Only the distance counts now.
- `simulate()` with some of a scene's sources or listeners (`sources=`, `listeners=`,
  and so `calibrate`) drew other random numbers for them than a run of the whole scene
  with the same seed, and others again when the names came in another order. Seeds go
  by the scene's order now, as in coverage maps; runs of the whole scene are unchanged.
- A negative `simulation.seed` or `--seed` passed every check, then stopped the run with
  `error: expected non-negative integer`. It is refused at once, naming the seed.
- `view --paths` and `export --paths` drew and counted every path up to the highest
  order within `ir_length_s`, where a simulation with `transition_ms` set, or with the
  statistical late method, keeps fewer as early paths (13 of the shoebox's 63 with the
  statistical method). They show those the simulation keeps now.
- Particles that reached their 200th reflection (in `animate` and `export --particles`
  longer than about 1 to 1.8 s) flew on through the walls. They stop there now.
- On a GPU, a late stage of more than about 1 to 3 million rays per band in a small
  room overflowed its sums, and was then traced all over again on the CPU. The GPU
  traces it now.
- When the GPU declined a run it had been given (its buffers too small for the
  results), the CPU traced it while the progress message said the GPU had. The message
  says the CPU now, and why; `auto` takes the CPU for such a run, where its trial
  stopped the run before. With `both`, a second CPU tracer took the GPU's parts beside
  the first, on the same threads; the one CPU tracer takes them over now. `tune` refuses
  such a scene, naming the reason, where it timed the CPU as the GPU, and
  `engine.ray_device.trace` returns the plan it followed with the results.
- The leak finder's area for a gap whose open edges do not close up into loops, or for
  rays slipping between faces, came out too small wherever rays escaped elsewhere too
  (by 23 % for two gaps of 30 m2 in a model of 100 m2). It is the gap's share of the
  reflections times the area of the surfaces now.
- `engine.wave.wave_responses` took `max_band_hz` up to 8000 Hz, as its error message
  said, and failed with an `IndexError` at 8000. Like the scene, it takes 63, 125, 250
  or 500 Hz now (`WAVE_BANDS_HZ`).
- `compare_parameters` compares IACC_E and IACC_L when both sets have them, with the
  just noticeable difference of 0.075 that ISO 3382-1 gives in annex B (the module's
  comment said the standard gave none).
- A Collada file with `<up_axis>X_UP</up_axis>` was read as if it were +z up, lying on
  its side, and `up_axis` could not correct it. It is turned upright now, and an
  `<up_axis>` that is none of X_UP, Y_UP and Z_UP is refused, where it was read as
  +z up.
- `calibrate(bounds=(0.0, high))` stopped with an error from scipy about infs or NaNs
  whenever a fitted material absorbed 0.01 or less in a band, as the built-in brickwork
  and concrete do at low frequencies. It refuses a lower bound of 0 at once, saying that
  the bounds must be 0 < low < high < 1.
- `acoustic-spaces inspect` of a model whose walls are drawn as a closed shell inside
  another gave the volume of the cavity between them, while the reverberation times
  below it used the air inside. It also gives the "air volume" now.
- A walk with places less than 10 ms apart (a `walkthrough --step` of about 1 cm at
  walking pace, or a walker faster than 25 m/s) stopped with an error about late parts,
  or with `--late-step 0` played some places through their neighbour's response: the
  places were named by their time to 10 ms, and shared names. They are numbered now
  (`<name> #<place> @ <time> s`). A walk whose points are all where it starts is
  refused, naming the walker, where it divided by zero.
- With the statistical late method every place of a walk has a late part of its own,
  but `WalkResponses.late_places` and the `walkthrough` summary said that only those
  every `--late-step` had. They say every place now.
- `walkthrough --hrtf` did nothing unless the format was binaural, and took a missing
  file without a word. Like `simulate --hrtf`, it now renders binaural through the HRTF
  when `--format` is not given, and refuses another `--format`.
- A coverage map over a slab whose top and underside are in one group (an IFC slab read
  without `space:`, or a solid floor from a CAD program) could put its points `height`
  above the underside, up to the slab's thickness too low. They stand on its top now.
- The STI of a coverage map point, and in `parameters.json`, was 0, or too low, where
  the decay of a band from 125 Hz up had not reached -20 dB by the end of the response
  (a short `ir_length_s` or `map --length`), while the other parameters there were
  undefined. It is undefined too now (`-`, or null in JSON).
- `validate`, `simulate`, `map` and the other commands that read a scene printed a
  Python traceback for a scene file that is not valid YAML, for a folder, and for an IFC
  or SketchUp model without its extra. They print an `error:` now, and the editor says
  why it could not open a file that is not valid YAML.
- A scene saved with an inline array whose element is a file (`array: {element: {file:
  cabinet.csv}, ...}`) kept that file's absolute path, so its folder could not move,
  and the editor's Save As with copies left the file behind. It is written relative to
  the scene's folder and copied along now.
- The editor's floor plan table wrote every corner back as it showed it, to six
  significant digits, when one changed or a corner was added or removed; the simulation
  form wrote back every setting when one changed, rounding the response length to 10 ms
  and the early-to-late time to whole milliseconds. They write only what changed now.
- A scene's `simulation.early.coherent_max_band_hz: null` (early paths in energy
  throughout) was left out when the scene was saved, and when the editor changed any
  other setting, so it came back as the default, 125 Hz. It is kept now. `Scene.to_data()`
  gives a scene as the data its file holds.
- The coverage form's note said the sources' sound power is spread evenly over the
  bands. It says how the map spreads it: as each source's spectrum, gain and equalizer
  set.
- The warning for edges that join faces of opposite orientation said that it only
  mattered for the volume. The volume and the inside checks do not depend on the
  normals; only their reported direction is unknown, and the warning says so.
- The documentation of `Calibration.uncertainty` and the `calibrate` table's `+-`
  column now says what it is, the standard error of the absorption for misses of one
  JND (how far it can move before the sum of the squared misses grows by one JND
  squared), and the module summary of `render` says the real-time convolver's
  partitions grow along the response, from the block size up to 8192 frames.
- The gallery example's comment said that its inner corner is where wall_4 and wall_5
  meet; it is wall_3 and wall_4.
- The user guide named the editor's menu entry **Edit > Add material** (it is **Add
  material from the database**) and said that arrays do not walk (only lines and
  surfaces do not; an array source walks like any point source), and gave the seminar
  room's 8 kHz reverberation time as 17 % short (18 %) in the octave bands.
- The validation report quoted the statistical late method's G as within 0.5 dB of the
  stochastic one (the test allows 0.7 dB), the BRAS octave analysis as within 4 % and
  0.5 dB of the published values (CR3's EDT at 500 Hz is 4.3 % lower, its C80 0.6 dB
  higher), the seminar room's 8 kHz shortfall as 17 % (18 %), the wave solver as
  leaving partitions out (it stops with an error) and the whole-room check as run with
  default settings (the wave solver is switched on). Its GPU timing table now says it
  comes from an earlier version of the script, and a paragraph that rendered as table
  rows is a paragraph again.
- CI runs the tests with `pytest -W error`, as the contributing guide said it did, so a
  warning fails the run; ruff runs on Linux.

## 3.0.0

Validation against two measured rooms, third-octave bands, vector-based scattering,
sources with their own spectra and line, surface and array sources, sound through walls
and between coupled rooms, fitting materials to measurements, ray tracing 7 to 8 times
faster, the late stage and the wave solver on a GPU, edge diffraction over two edges,
early reflections in pressure at low frequencies, IFC and Collada models, material
libraries, a leak finder and walkthroughs.

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
