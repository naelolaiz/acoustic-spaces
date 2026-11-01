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

## Terrain

The ground need not be flat: `ground.terrain` gives its heights on a square grid, from
a file or in the scene, and everything stands on it.

```yaml
geometry:
  ground:
    terrain: {file: hills.asc}        # an Esri ASCII grid, or x y z points on a grid
    map_origin: [552000, 5806000]     # the map's point that becomes the scene's (0, 0)
    sky: 60
```

- `terrain.file` is an Esri ASCII grid (`.asc`, what GIS tools write and
  `gdal_translate -of AAIGrid` makes of a GeoTIFF or any other raster) or a table of
  `x y z` points on a regular grid (`.xyz`, `.txt` or `.csv`). Its coordinates must be
  in the geometry's units on a plane (a projected system such as UTM), not degrees of
  longitude and latitude; `ground.map_origin` is subtracted from them, so that the
  scene's coordinates stay small, and from a buildings file's (see
  [Buildings from a map](#buildings-from-a-map)).
- `terrain.heights` gives the heights in the scene instead: rows of values, each row
  along x from `origin`, the rows along y, `spacing` apart.
- `ground.height` is added to the terrain's heights.

Between the grid's nodes the ground is bilinear; past the grid's edge it goes on at the
height of the edge. The scene's ground is the grid sampled over the ground's extent and
cut into triangles: the whole grid unless `extent` is set, which must then lie inside
it (the terrain sets the ground's size, not the margin). The sky is `sky` over the
terrain's highest point, or `margin` over the highest building, barrier or object.
Ground zones take the triangles whose centres lie in them. Buildings and barriers stand
on the terrain however it runs: a building's walls go down a little below the lowest
ground under them and its roof lies its height over the terrain's mean height at the
footprint's corners; a barrier's top follows the terrain along its path; vegetation
stands on the terrain under its middle. The model file's objects are taken as they are,
and one that reaches below the terrain's lowest point is refused. A terrain with wind or
temperature gradients is not supported yet: the scene is refused.

`acoustic-spaces inspect` describes the terrain (its points, its heights and its grid)
and gives each source's and listener's height over the ground under it. Coverage maps
follow the terrain: their points stand at the map's height over the ground under them.

### How the simulation takes the terrain into account

- **The late rays** bounce off the terrain's triangles as off any surface.
- **The early paths** do not reflect off its triangles: a facet a few metres across
  would send a 5 m wave back as a plain does, and as a listener moves the reflections
  off neighbouring facets would come and go in jumps. In sight of the source the ground
  reflects once instead, off the **mean plane** of the terrain's profile between the
  source and the listener, as the outdoor standards (CNOSSOS-EU, Nord2000) have it: the
  plane through the least-squares line of the profile, level across it, with the
  reflection factor of the ground under the reflection point (its material, or the
  grounds of the zones its Fresnel zone covers, as on flat ground). Over a plain or a
  plane slope that is the ground itself, and the reflection is the one flat ground
  gives.
- **Crests** between the source and the listener bend the sound over them. The ground's
  profile in the vertical plane between the two is drawn, and the string pulled taut
  over it from one to the other bends at the crests in the way: each is a wedge whose
  faces run from it down to the ground under its neighbours on the string (the
  source's foot, the listener's or the next crest), so a ridge of two slopes is the
  wedge it is and a rounded hill the wedge of its chords. The sound over them is the
  uniform theory of diffraction's, wedge after wedge: the direct wave over each edge
  and the ground's reflections before and after it, off the profile's segments between
  the crest and its neighbours on the string, each weighted by the share of its
  Fresnel zone the segment holds (Nord2000's Fresnel-zone weights, so that at low
  frequencies they add up to the one reflection off the side's mean plane and at high
  ones the segments that mirror the sound to the edge carry it all), with the ground's
  reflection factor (the spherical wave's over a ground with an impedance). Near the
  crest a reflection is a term of the wedge's coefficient, as in the exact solution;
  far from it, a path of its own, as in the four-path model of a barrier, and the
  uniform theory's transition over its extra path bridges the two. Where the string
  runs along the ground over a flat or rounded top, the wedge sits where the string's
  two legs onto it would
  meet above it (the equivalent edge of a thick screen), and the crest screens more the
  wider it is, by ISO 9613-2's factor for a thick screen (up to 4.8 dB more at the
  highest bands): its width is the ground within an eighth of a wavelength of the
  string, less what a sharp wedge would have, so a ridge has none and a rounded hill a
  few metres at the highest bands. The path over the crests arrives from the last crest
  after the string's length, with the air's attenuation over it.
- **In sight of the source**, the crest the ground comes nearest the line of sight
  still diffracts: its wedge's waves are added to the direct sound and to the
  reflection off the slope before it (where the listener sees the source in that
  slope), with the uniform theory's transition at each boundary, so the level is
  continuous across the edge of a hill's shadow and ripples in the lit zone as the
  exact field does. The mean plane's reflection takes over from the slope's as the
  crest drops below the line of sight, by half the longest wavelength of path
  difference.
- Buildings, barriers or model objects standing on the string stop the path: the sound
  bent over a hill and a building in turn is not modelled. Edges buried in the terrain
  do not diffract, nor do its own ridges, which bend too little at each grid line: the
  sound over the crests takes their place.

A ridge agrees with the exact solution for the wedge within 0.2 dB, a flat terrain and
a plane slope with the flat ground and the image source, and a smooth hill over a soft
ground lets 1.4 to 1.8 dB more over its crest than the thin screen's attenuation (see
[Validation](validation.md#terrain-against-closed-forms-and-the-exact-wedge)); a
rounded crest, which the sound creeps over, has no exact solution, and is an
engineering model here as in the standards.

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

## Buildings from a map

`geometry.buildings_file` adds the buildings of a GeoJSON file (`.geojson` or `.json`)
to the scene's own: each `Polygon` feature, and each part of a `MultiPolygon`, is a
building with its outer ring as the footprint and its height in a property.

```yaml
geometry:
  ground:
    map_origin: [552000, 5806000]
  buildings_file:
    file: town.geojson
    height: height          # the property with each building's height, or one height for all
    default_height: 9       # for buildings whose property is missing
    name: name              # the property with each building's name; else building_<n>
    without: [old_mill]     # buildings of the file to leave out, by name
```

The coordinates must be in the geometry's units on a plane (a projected system such as
UTM); a file in degrees of longitude and latitude is refused, with a note to project
it. `ground.map_origin` is subtracted from them, as from a terrain's. Holes in a
footprint (courtyards) are built over, and the report says which footprints had them.
The file's buildings come after the scene's own in the lists, in `inspect` and in the
editor's buildings table, where **Remove** leaves one out (it goes into `without`), and
they count for insertion losses like any other building.

A whole town or a district from a map runs as it is. Edge diffraction looks for paths
that reflect off a wall, a roof or the ground before or after bending over an edge, and
it skips the walls and roofs that no such path can reach before it traces a ray, with
the same results as testing them all. On four cores 900 buildings simulate in about 17
s and 3,600 in about 56 s (see
[Validation](validation.md#edge-diffraction-in-whole-cities)).
About five sixths of that time for 900, and over half for 3,600, goes to the terms that
make reflections fade at the edges ([Simulation](simulation.md#edge-diffraction)).
The work still grows with the square of the buildings: for tens of thousands, cut the
map to the area round the sources and listeners, or set
`simulation.early.diffraction: direct`, whose cost grows with the edges alone.
`second_order` looks at every pair of edges that see each other, which in a city is
most of them: keep it for a street or a site. A coverage map pays the cost of a
listener at each of its points.

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
  powder:
    impedance: {flow_resistivity: 5, reaction: extended}  # the impedance changes with the angle
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
- `reaction` is how the ground meets sound at an angle: `local` (the default), the same
  impedance at every angle, right for grass and harder grounds; or `extended`, sound
  also travelling sideways inside the ground, so that a reflection meets the impedance
  of its own angle, Zc n / sqrt(n² - sin² θ) with n the ground's complex index (and the
  layer's with it; Attenborough 1985), which tells for soft snow, moss, forest floors
  and thin layers: a layer of snow 10 cm deep on ice reflects up to 3.7 dB differently
  20 m away, fresh snow 1.0 dB, grass 0.1 dB. The early paths take it at each
  reflection's angle and the parabolic equation at grazing incidence; the wave solver,
  whose ground is one impedance for all angles, reacts locally. Not with
  `two_parameter`, which has no index (see
  [Validation](validation.md#ground-reflection-against-the-exact-solution)).
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
scene unless [weather zones](#weather-zones) say otherwise: this suits open ground
better than streets, where buildings shelter the wind and channel it. See
[`examples/open_air_stage/scene.yaml`](https://github.com/naelolaiz/acoustic-spaces/blob/main/examples/open_air_stage/scene.yaml):
a stage heard 100 and 300 m away across a park, downwind and upwind.

The 3D view (the editor's, and `acoustic-spaces view`) draws the wind: sky blue arrows
over the ground, at the height the wind's speed is given at (`wind.height_m`),
pointing the way it blows and as long as it is strong, the strongest a little shorter
than the space between them. The editor's plan draws the same arrows from above, with
the scene's wind in its top right corner (`wind 5.0 m/s at 10 m`). Arrows keep clear of
the sources and listeners, and none stand inside a building taller than them. In the
editor, **View > Show wind profile** also stands a mast in each weather where the wind
blows, the scene's and each zone's: arrows at heights evenly up to the wind's height
and a curve through the wind at every height between, how it grows from the still air
at the ground; **Show the profile** plots the same in numbers.

### Weather zones

The weather is not the same everywhere: the wind drops in the lee of a row of
buildings or a wood, the air lies cooler over a lake and warmer over a car park, and a
sea breeze blows along the shore and dies away inland. `atmosphere.zones` draws parts
of the scene with weather of their own:

```yaml
atmosphere:
  preset: windy                      # 5 m/s 10 m up, blowing towards +x
  zones:
    - name: lee                      # behind the warehouses the wind drops
      polygon: [[120, -60], [300, -60], [300, 60], [120, 60]]
      wind: {speed_m_s: 1}
    - name: lake                     # cooler air over the water, under an inversion
      polygon: [[300, -200], [600, -200], [600, 200], [300, 200]]
      preset: clear_night
      temperature_c: 12
```

A zone is a polygon in the plan, in the geometry's units, with a `name` (numbered
when left out) and any of the atmosphere's fields: `preset`, `wind` (its
`speed_m_s`, `direction_deg`, `height_m` and `profile`), `temperature_gradient_k_m`,
`temperature_log_k`, `roughness_m` and `table`, and `temperature_c`, the temperature
at the ground there. What a zone leaves out it takes from the scene's atmosphere: a
zone with `wind: {speed_m_s: 1}` alone has the scene's wind slowed to 1 m/s, blowing
the scene's way, with the scene's temperature profile and roughness. A `preset` in a
zone starts from that preset instead, its wind blowing the scene's way unless the zone
gives `direction_deg`, and the zone's other fields change it; a zone's `table` replaces
the formulas, and a zone's gradient or log term drops the scene's table. Outside the
zones the scene's weather holds, and where zones overlap the first wins. The eddies
(`turbulence`) are the scene's everywhere. A zone that restates the scene's weather
changes nothing but rounding. `inspect` names each zone, its corners and its weather;
in the editor, **Weather of** chooses between the scene's weather and each zone's, and
**Add weather zone** draws one in the plan (see [the editor](editor.md#outdoor-scenes)). The
views draw each zone with its name and its wind's speed at the scene's wind height
(`lee: 1.0 m/s`), with its ground temperature when it has one of its own
(`lake: 0.0 m/s, 12.0 °C`), and its own arrows in it; the 3D view raises see-through
walls round it up to the wind's height.
[`examples/lakeside_concert/scene.yaml`](https://github.com/naelolaiz/acoustic-spaces/blob/main/examples/lakeside_concert/scene.yaml)
has two: a concert on a lawn by a lake, heard across it, with cool air and a stronger
breeze over the water and a weaker breeze among the houses on the far shore.

A zone's border is a step: the wind does not die away gradually into the lee. How
each stage of the simulation takes the zones is described below; the parabolic
equation and the wave solver go through them, and the early paths, whose arcs need one
profile, take the mean along their way.

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
  quarter period. Against an independent ray tracer on the smooth profile of the
  `windy` preset downwind, the levels of three such paths 100 and 300 m away are within
  0.05 dB (the tests allow 0.2 dB); no such check is made near a caustic.
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
  rays give it. In still air the parabolic equation is within 0.2 dB of the exact
  solution over grass and harder grounds from a source 1 m up, 25 to 400 m away, and
  within 0.45 dB from a wind turbine's hub 178 m away (see
  [Validation](validation.md#ground-reflection-against-the-exact-solution)). In wind,
  against a fast field program on the 1995 benchmark cases with the gradient stopping
  100 m up, as the wind's does, it is within 0.35 dB to 1 km downwind, and at 10 km
  within 3 dB at 100 Hz and 11 dB at 1 kHz (47 dB down); upwind within 0.1 dB until the
  sound is 20 dB down and up to 6 dB off deeper into the shadow. A gradient that goes on
  to a kilometre up bends sound back at angles it does not carry, and beyond 1 to 2 km
  its pure tones stray there by 3 to 20 dB, though the bands a scene reports stay within
  0.8 dB of the program to 2 km (see
  [Validation](validation.md#outdoor-sound-in-wind-against-a-fast-field-program)).
- **The wide-angle parabolic equation** takes its place with
  `simulation.early.parabolic_equation: wide_angle` (`--parabolic-equation wide_angle`
  for one run of `simulate` or `map`, **Parabolic equation** in the editor). The
  default, `narrow_angle`, takes the refraction as a phase shift at each step, which
  holds for sound within some 15 degrees of the horizontal and a speed of sound close to
  the ground's. The wide-angle one (split-step Padé, as in Collins' RAM) marches the
  field with the exact one-way operator, written as a rational function of ten terms,
  on finite differences of fourth order in height, with the ground's impedance as its
  boundary and a starting field of the operator's own; it carries sound at up to 45
  degrees from the horizontal through air of any speed. On the benchmark's gradients
  going on to a kilometre up it is within 0.11 dB of the fast field program to 10 km
  (0.4 dB at 10 Hz 44 dB deep in a shadow) where the narrow-angle one strays by 3 to
  20 dB, and in the weather presets within 0.02 dB per band on average where the
  narrow-angle one is within 0.08 dB (see
  [Validation](validation.md#the-wide-angle-parabolic-equation)). It takes three
  to four times the work, all of it in the bands the parabolic equation gives.
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
  leaves. The CPU and the GPU trace these rays alike (`simulation.late.device`): the
  GPU follows the same chords, in single precision, and what reaches each listener in
  each band, and when, agrees within the noise of the ray count (see
  [Validation](validation.md#outdoor-scenes-on-the-gpu)). On Direct3D 12 (Windows) the
  CPU traces them in a scene with vegetation too, which the run says.
- **The wave solver** (`simulation.wave`) steps the sound field itself in its bands, with
  a speed of sound that changes from cell to cell: at each cell's height, the still
  air's plus the wind's component along the horizontal from the source, as above, so
  the rays, the parabolic equation and the wave solver bend sound alike. Its grid is
  fine enough for the slowest sound and steps in time for the fastest, a few percent
  more cells and steps than in still air in the `windy` preset. Over grass, 50 to 200
  m from the source, in the 63 and 125 Hz bands, it is within 0.7 dB of the parabolic
  equation, 0.2 dB on average (see [Validation](validation.md#the-wave-solver-in-wind-and-temperature-against-a-parabolic-equation)).
  It runs on the CPU or on the GPU alike (`simulation.wave.device`), the two agreeing
  to rounding. The run's details give the slowest and the fastest speed of sound on the
  grid (`wave_speed_min_m_s`, `wave_speed_max_m_s`).
- **Moving air in the wave solver.** The effective speed of sound gives each cell one
  speed, whichever way the sound crosses it: sound that a façade downwind sends back
  towards the source goes at the speed of sound going away from it. With
  `simulation.wave.wind: moving` (`simulate --wave-wind moving`, or **Wind** on the
  editor's Simulation page) the air moves instead: each cell keeps the still air's
  speed of sound, and the wind carries the sound along with it, so that sound goes at
  c + v downwind and c − v upwind, whichever way it turned. In a uniform wind of 15
  m/s it is within 0.05 dB and 0.1 ms of the exact solution in every direction, where
  the effective speed of sound is up to 0.9 dB off; the echo of a façade 40 m
  downwind comes within 0.02 ms of ray theory's, where the effective speed of sound
  brings it 10.6 ms early; in the `windy` preset over grass it is within 0.34 dB of the
  parabolic equation, where the effective speed of sound is within 0.69 dB (see
  [Validation](validation.md#moving-air-in-the-wave-solver)).
  It costs three to four times as much as the effective speed of sound, on the CPU only
  (asked for the GPU, it says so and stays on the CPU), and the run's details say
  `wave_moving_air`. Its grid is fine enough for sound going against the fastest wind
  in the slowest air.
- **Through weather zones** the parabolic equation marches zone by zone, changing its
  speed of sound where its path crosses a border, and the wave solver takes each cell's
  zone. The early paths, whose arcs need one profile, take the mean profile along their
  way, each zone's weighted by the length of the way over it (the equivalent profile
  of Nord2000 and Harmonoise), with the wind's component along the way from each
  zone's own direction; so do the paths off façades and barriers, over their legs. The
  late rays bend through the mean weather over the scene's extent, each zone's
  weighted by its area. Against a Crank-Nicholson parabolic equation marched through
  the zones, the engine's parabolic equation is within 0.15 dB on average and 1.2 dB
  at most to 800 m, over shelter and gusts downwind and upwind, an inversion past a
  lake shore and cool water downwind. The mean profile, at the bands the rays give,
  is within 0.35 dB over the cool water; where the wind changes, or an inversion sets
  in, it is up to 6.6 dB off (at 2 kHz 800 m downwind of a shelter), mostly giving 1
  to 4 dB more sound and in a few places up to 2.7 dB less: one profile cannot both
  bend the sound down over the first 150 m and let it go straight after, the cost of
  the equivalent profile the standards take (see
  [Validation](validation.md#weather-zones-against-a-crank-nicholson-parabolic-equation)).
- **The animation's particles** and **the drawn paths** (`view --paths`, the Blender
  export) curve as the sound does.

Bending takes time: the `open_air_stage` example runs in 3 s instead of 0.2 s in still
air, a scene's late rays take about three times as long, and coverage maps about six
times, the parabolic equation running for each point; `simulation.performance.threads`
shares the work out over the cores. The places of a walking source share the parabolic
equation's runs where they stand at one height over one ground and neither the weather
nor the ground has zones: a run from one place serves the listeners of every other that
the sound goes the same way to, each reached by a last step of its own, so that it gets
what a run of its own place would give it; only the absorbing layer can lie higher,
where the farthest of the run's listeners needs it. An aircraft's 26 places 64 m apart,
100 m up over 1.6 km of a meadow in the `windy` preset, take 31 s instead of 89 s on
four cores, with the same early levels to 10⁻⁷ dB. An atmosphere in which sound goes straight (`neutral`, no wind
and no temperature gradient) changes nothing: the results are those of still air, bit
for bit.

**Without turbulence, shadows come out deeper than measured.** The eddies of the wind
and of the temperature scatter sound into a shadow: in the validation's shadows, 280 to
500 m out and from 500 Hz up, they keep it 5 to 25 dB below the free field, where
without them it is 50 to 70 dB deep. Give the atmosphere its `turbulence` (see [Turbulence](#turbulence)).

Against measurements of a wind turbine up to 845 m away in ten spells of different
weather, from light winds to strong upwind, the bands from 80 Hz to 2 kHz are within
1.7 to 2.7 dB on average, about as close as the published parabolic equation model in
the eight cases other than strong upwind, and closer in those two (see [Validation](validation.md#outdoor-propagation-against-measurements-hannover)).

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
  stays some 5 to 25 dB below the free field instead of 50 to 70 (see
  [Validation](validation.md#turbulence-against-a-parabolic-equation-through-random-eddies)).
- The sound of a far source **flickers**, louder and softer by several decibels from
  one second to the next, as the eddies drift past.

```yaml
atmosphere:
  preset: windy
  turbulence: moderate          # weak, moderate or strong; or the values themselves:
  # turbulence: {temperature_structure: 0.008, wind_structure: 0.12, outer_scale_m: 10}
  # turbulence: {preset: moderate, height_m: 2}   # the strength at 2 m, weaker higher up
```

`turbulence` goes in the `atmosphere`, also in still air (`atmosphere: {turbulence:
strong}`). Its fields:

| field | what it sets |
|---|---|
| `preset` | a strength to start from (below); fields given with it change it |
| `temperature_structure` | the temperature's structure parameter C_T² (K² m^-2/3): about 0.001 to 1 near the ground, the most in sunshine |
| `wind_structure` | the wind's structure parameter C_v² (m^4/3 s^-2): about 0.01 to 1 near the ground, the more the stronger the wind |
| `outer_scale_m` | the size of the largest eddies (m, 10 by default): some metres near the ground |
| `height_m` | the height above the ground (m) the structure parameters hold at, from which the eddies weaken with height as (z / `height_m`)^-`falloff`, taken no nearer the ground than `outer_scale_m`; none by default: the same at every height |
| `falloff` | how fast they weaken with height, the power of z / `height_m` (2/3 by default): 2/3 in a neutral or stable surface layer, up to 4/3 in strong sunshine |

| preset | C_T² | C_v² | weather |
|---|---:|---:|---|
| `weak` | 0.001 | 0.03 | overcast, a light wind |
| `moderate` | 0.008 | 0.12 | a breeze and some sun |
| `strong` | 0.08 | 0.5 | a strong wind or a hot sun |

These are indicative, for the strength near the ground, and are taken as the same at
every height unless `height_m` is given. Real eddies are strongest near the ground and
weaken with height: in the surface layer the structure parameters fall as z^(-2/3) in
neutral and stable air (the wind's always, the temperature's as z^(-4/3) when the sun
drives the air; Monin-Obukhov similarity, Wyngaard, Izumi and Collins 1971). With
`height_m` the values hold at that height, where they were measured or estimated
(usually 2 m), and fall above it by that power; nearer the ground than the eddies'
outer scale the engine takes them as at the outer scale (with `height_m` 2 m and the
default 10 m, 0.34 of the given values up to 10 m, then falling), because eddies that
size reach that high and the thin layer of stronger ones right at the ground does not
tell in the parabolic equation (see [Validation](validation.md#turbulence-against-a-parabolic-equation-through-random-eddies)).
Without `turbulence`, or with both structure parameters 0, the results are those
without eddies, bit for bit.

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
- **Eddies weaker with height** (`height_m`) enter where each pair of paths runs: the
  phase difference two paths gather over a stretch grows with Cn² at the mean height of
  the two points there, so paths aloft part slower than with eddies the same everywhere;
  the sound scattered into a shadow takes Cn² at the height it scatters at, and the
  shaking of the refraction Cn² at the geometric mean of the source's and the listener's
  heights; all of them no nearer the ground than the eddies' outer scale. Against the
  parabolic equation through eddies strengthening right down to the ground, the ground
  dip is within 0.9 dB on average over the soft grounds and 1.5 over the harder ones
  (0.8 and 1.0 with eddies the same at every height) and the shadows within 1.3 to
  2.9 dB (0.7 to 1.2); taking Cn² at the mean height itself, right down to the ground,
  would put the dip 1.7 dB off over the soft grounds (see
  [Validation](validation.md#turbulence-against-a-parabolic-equation-through-random-eddies)).
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
  wind's speed, at least 1 m/s; at most every 50 ms). Each early path plays through its
  own delay, which goes from one moment's to the next's (its top band's; how its other
  bands lag behind it goes into its filter), as along a
  [walk with Doppler](spatial-audio.md#moving-sources-and-doppler): crossfading the
  moments' responses instead would mix each path at two delays a few microseconds
  apart, which cancels the treble half-way between them. The moments change as an
  Ornstein-Uhlenbeck process, so the delays drift rather than jump, and a walker keeps
  the same eddies from place to place. The `auralize` command
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
| `{leaf_area_per_m3: 0.5, leaf_width: 0.06}` | the leaves themselves: their area (one side's) per cubic metre of the volume and their width (the diameter of a disc of a leaf's area, m); what they scatter and absorb in each band, and how far forward they scatter it, follow from their size (below) |

ISO 9613-2's rates are what dense foliage takes from sound passing through it in all,
with what it scatters onwards counted in, so the preset absorbs all of it and adds no
scattered sound. The standard gives them for 20 to 200 m of foliage, a fixed 1 to 3 dB
for 10 to 20 m and none below; the engine takes the rates per metre at any depth, so a
belt under 20 m deep loses a little less than the standard says, and it measures the
depth along the straight path, where the standard takes an arc of 5 km radius.

**Leaves** given by their area and width are taken as thin rigid discs of that size,
turned every way alike, `leaf_area_per_m3` of them per cubic metre. Each scatters as
a disc does: small against the wavelength, as a dipole, from the air it holds back
(16 (kr)⁴ / (81 π²) of its area, k the wavenumber and r its radius), so the low bands
pass almost untouched; large against it, twice its mean projected area, half of that
the shadow it casts, which goes nearly straight on, and half reflected off its faces,
which go every way alike. In between the two limits are bridged smoothly, as the
exact cross-section of a disc about a wavelength across is not in closed form. The
leaves absorb in the viscous and thermal boundary layers on their two faces, a few
per mille of their area, which is what they take at the low frequencies where they
scatter nothing. Per metre, the volume takes its leaf area times those
cross-sections over a disc's area, averaged over each band (the speed of sound sets
the wavelength); the scattered share and its mean cosine (0 for the dipole, towards
1/2 for a large disc) replace `albedo` and `asymmetry`, which leaves do not take.
Leaves 5 cm wide at 0.5 m² per m³, a dense canopy, take 0.05 dB per metre at 2 kHz,
0.5 at 4 kHz and 1.6 at 8 kHz out of the early paths, and almost nothing below 1 kHz;
a hedge's leaves (2 to 6 m² per m³) take several decibels per metre from 4 kHz up. Much of it
comes back scattered, with the late rays, so what a listener loses is far less
(see [Validation](validation.md#leaves-against-closed-forms-and-iso-9613-2)). A real
leaf is neither rigid nor a disc, and leaves hang in clumps: measured foliage, where
there is any, is the better input, as `attenuation` with an `albedo` and an
`asymmetry`.

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
  [Validation](validation.md#vegetation-against-a-particle-simulation)). The CPU and
  the GPU trace rays through vegetation alike; vegetation of more than 16 media (kinds
  of foliage and stems in one scene), and on Direct3D 12 (Windows) vegetation in wind
  or temperature that bends the rays, is traced on the CPU, which the run says.
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
each, foliage (or leaves) and stems together. The editor's 3D view, `view` and the export to
Blender draw the volumes in see-through green, and coverage maps outline them.

## What the simulation does outdoors

- **Early paths** are found as in a room: the direct sound, reflections off the ground,
  façades and barriers, and (with `simulation.early.diffraction`) the sound bent over
  roofs, round the corners of buildings and over the top and round the ends of barriers.
  Edges on the ground (the foot of a wall or a barrier) and the inside corners where two
  wings of a building meet diffract nothing. `second_order` follows the sound over a
  building, its front and back edges in turn; `first_order` bends over one edge at a
  time, which is enough for a barrier. With `diffraction_reflections: both_sides` a
  path reflects off the ground both before and after the edge, the fourth of the four
  paths over a barrier on the ground (see
  [Edge diffraction](simulation.md#edge-diffraction)).
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
- **A terrain** reflects the early sound off the mean plane of its profile and bends
  it over its crests, and the late rays off its triangles (see
  [How the simulation takes the terrain into account](#how-the-simulation-takes-the-terrain-into-account)).
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

## Levels over a long term

The weather changes from day to day, and so does the level a road or a wind farm makes
a kilometre away: noise maps and assessments give the level over a long term, a year,
in which each weather counts for the share of the time it holds (ISO 1996-2,
CNOSSOS-EU, Nord2000). `long_term` gives the weather classes and their shares:

```yaml
atmosphere:
  preset: windy                       # 5 m/s 10 m up, towards the listeners
long_term:
  classes:
    - name: downwind                  # the scene's own weather
      share: {day: 0.3, evening: 0.25, night: 0.15}
    - name: upwind
      wind: {direction_deg: 180}
      share: {day: 0.3, evening: 0.25, night: 0.15}
    - name: still
      preset: neutral
      share: {day: 0.4, evening: 0.3, night: 0.2}
    - name: inversion                 # clear nights: the air warms with height
      preset: clear_night
      temperature_c: 8
      turbulence: weak
      share: {day: 0, evening: 0.2, night: 0.5}
```

A class has a `name` (`class_1`, `class_2` and so on when left out), its `share` of the
time, and any of a weather zone's fields over the scene's weather: `preset`, `wind`, the
temperature's gradient and log term, `roughness_m`, `table` and `temperature_c`, the
ground's temperature. They resolve as a zone's do (see [Weather zones](#weather-zones)),
and a class without them is the scene's own weather. A class may also have
`turbulence`, eddies of its own in place of the scene's: weak on a clear night, strong
in the sun. The weather zones hold in every class. `share` is one number for the whole
day, or one per period of the day, and in each period the classes' shares add up to 1.

The periods are by default the day (12 hours), the evening (4 hours, 5 dB added) and
the night (8 hours, 10 dB added) of L_den (Directive 2002/49/EC); `long_term.periods`
gives others, each with its `name`, `hours` and `penalty_db`:

```yaml
long_term:
  periods:                            # 15 hours of day, 9 of night with 10 dB (L_dn)
    - {name: day, hours: 15}
    - {name: night, hours: 9, penalty_db: 10}
  classes:
    - {name: downwind, share: {day: 0.5, night: 0.3}}
    - {name: inversion, preset: clear_night, share: {day: 0.5, night: 0.7}}
```

The scene is run as it is, for its impulse responses, and once in each class, all with
the same seed so that every run traces the same rays; a class whose weather is the
scene's own takes the scene's run, so n classes cost at most n + 1 runs. Then, at every
listener and every map point, per octave band and A-weighted:

- the level in each class;
- the level in each period: the energy mean of the classes' levels, each by its share
  of that period, L = 10 lg Σ p_k 10^(L_k/10);
- the **long-term level**: the energy mean of the periods' levels by their hours;
- the **rating level**: each period's A-weighted level with its penalty added, averaged
  by the hours. With the default periods it is L_den,
  10 lg ((12·10^(L_day/10) + 4·10^((L_evening + 5)/10) + 8·10^((L_night + 10)/10)) / 24);
  with others, L_R, a whole-day rating level as ISO 1996-1 defines it.

The long-term level relative to free field, and the long-term insertion loss of what
`insertion_loss_of` names (taken in every class), come with them. `simulate` prints
them. `parameters.json` has a `long_term` block (the classes, their shares and the
periods), and each response a `long_term` entry with `spl_db`, `spl_a_db`,
`rating_level_db` and `rating` (L_den or L_R), the levels of each period and each
class, and `re_free_field_db` and `insertion_loss_db` with their A-weighted values.
Maps gain the parameters `long_term` (dB per band) and `rating_level` (dB(A)).
`inspect` lists the classes with their shares and weathers, and the periods. From
Python, `acoustic_spaces.engine.long_term.simulate` and `coverage_map` take
`long_term=` to run classes the scene does not give, and `Scene.under(weather_class)`
is the scene in one class. In the editor, weather classes are added and changed under
**Weather of** (see [the editor](editor.md#outdoor-scenes)).

The level in each class is as accurate as the simulation in that weather (see [How the
simulation bends sound](#how-the-simulation-bends-sound)). The combination is the
standards' formula, exact to rounding; over grass, the correction it gives from the
classes favourable to propagation to the long-term level follows ISO 9613-2's within
0.8 dB, with a C_0 of 3.4 dB, inside the standard's 0 to 5 dB (see
[Validation](validation.md#levels-over-a-long-term)).

## Passing vehicles

A car, a train or an aircraft is a source with a [walk](scenes.md#walks) and `doppler:
true`: `walkthrough` plays it past the listener with its pitch shifting along each
path, and prints the listener's L_Amax, sound exposure level and L_Aeq over the
pass-by; `--levels pass.csv` saves the level over time, A-weighted and per octave
band (see [Moving sources and Doppler](spatial-audio.md#moving-sources-and-doppler)).
Over a hard ground, the sound is within 45 dB of the exact field at 30 m/s (see
[Validation](validation.md#moving-sources-against-the-exact-field)). A walking
listener passes standing sources the same way.

### Helicopters

From Python, `acoustic_spaces.rotorcraft` gives the sound a helicopter's two rotors
send each way, as an `Emission` for a walk with Doppler (see
[Moving sources and Doppler](spatial-audio.md#moving-sources-and-doppler)):

- `trim` balances a `Helicopter` flying level at an airspeed by momentum theory: the
  main rotor's thrust holds up its weight and holds back its fuselage's drag, the disc
  tilted forward between them; Glauert's inflow gives the induced velocity through the
  disc, and the tail rotor's thrust holds the main rotor's torque.
- `emission` then gives each rotor's **tones**, the harmonics of its blade passing
  frequency from its blades' steady lift, drag and thickness (the Ffowcs Williams and
  Hawkings equation far from the rotor, Farassat's formulation 1A, up to the harmonic
  90 dB under the loudest), and each rotor's **broadband noise**, louder ahead and
  quieter behind as a moving source is. It is the pressure 1 m from the main rotor's
  hub, in pascals, as the far field has it, in the helicopter's frame: x along its
  flight, y to its left, z up; a walk with `face_forward` turns it along its path.
- The broadband noise comes from one of two models, the helicopter's `broadband`:
  `"pegg"`, the default, is Pegg's semi-empirical method (NASA TM-80200), a fit to
  measured rotors' noise, one spectrum for each rotor round a peak set by its thrust
  and its tip speed. `"bpm"` is each blade section's own noise in the flow past it,
  by Brooks, Pope and Marcolini's airfoil self-noise (NASA RP-1218,
  `acoustic_spaces.airfoil_noise`): the turbulent boundary layer at the trailing edge,
  the laminar one's vortex shedding, the trailing edge's bluntness and the tip's
  vortex, for 24 sections of each blade at 144 places round the turn, each at its own
  speed and angle of attack from the blades' trimmed pitch (`rotorcraft.pitch`), its
  bands moved by its own Doppler shift. A `Rotor` gives the blades' twist, lift
  slope, trailing edge and tips, and whether their boundary layers are tripped.
  `rotorcraft.broadband(trim, ..., model="bpm")` picks one model for a call.
- `AS350_B3` is an Airbus Helicopters AS350 B3 at its maximum mass of 2,250 kg, from
  its type certificate and NASA's flight tests of the type; its main blades' chord
  and twist and its tail rotor's speed, chord and thickness are assumed (the source
  code says which is which).

```python
from acoustic_spaces import rotorcraft
from acoustic_spaces.walkthrough import render_walk

trim = rotorcraft.trim(rotorcraft.AS350_B3, 63.0, temperature_c=25.0)  # 63 m/s
sent = rotorcraft.emission(trim, seconds=30.0)  # 48 kHz: tones and broadband noise
sound = render_walk(walk, {"helicopter": sent})  # its walk with doppler: true
```

The tones agree with Gutin's formula for a hovering rotor and with Farassat's
formulation in time, blade point by blade point, to within 10⁻⁵ of the loudest
harmonic; Pegg's worked example comes out as he gives it, and the airfoil self-noise
matches the two sample outputs of its authors' code to within 0.01 dB (see
[Validation](validation.md#helicopter-rotors-against-theory-and-a-certified-level)).
Flying the AS350 B3 over a microphone as its noise certification does, 150 m up at
63 m/s, the two models fall on either side of the certified 84.6 dB(A) of sound
exposure:

| broadband noise | sound exposure level | against the certified 84.6 dB(A) | main rotor's broadband alone |
|---|---|---|---|
| Pegg's (`"pegg"`, the default) | 88.0 dB(A) | 3.4 dB over | 87.3 dB(A) |
| each blade section's own (`"bpm"`) | 80.1 dB(A) | 4.5 dB under | 68.6 dB(A) |

With Pegg's noise the main rotor's broadband noise is nearly all of it. Pegg's method
is a correlation of the 1970s' measurements; in his own worked example the band of the
broadband peak comes out 5 dB over the level measured there, though a tone of the tail
rotor falls in that band too. The blade sections' own noise is 19 dB under Pegg's for
the main rotor, and the tail rotor's tones (79.8 dB(A) on their own) are then nearly all of it: self-noise
is the noise of a blade in a smooth flow, and leaves out the blades cutting through the
wake of those ahead and through turbulence, which a full-size rotor in flight makes
and which Pegg's fit to measured rotors holds.

A scene's source carries the sound power and the directivity of what the rotors send,
for the levels the simulation and the pass-by report and for the late part:
`sent.sound_power_db(trim.density, trim.sound_speed)` gives the one per octave band
and `sent.directivity()` the other, which
`acoustic_spaces.directivity.write_directivity_csv` writes as a table a scene reads
(`directivity: {file: helicopter.csv}`). The `helicopter_flyover` example has them for
an AS350 B3 150 m up at 63 m/s against a light wind, and its
[flyover demo](demos.md#a-helicopter-flying-over) flies it over a meadow, its rotors
trimmed at its airspeed: its speed over the ground less the wind at its height.

What it leaves out: loads that change round the turn (in forward flight the
advancing blade carries less lift than the retreating one, and the blades flap), the
blades meeting the tip vortices of those ahead (blade-vortex interaction, the slap
of a descent or a turn), the engine and the gearbox, and the fuselage's reflection and
shadow. `trim` takes the airspeed it is given, through still air: the flyover demo
gives it the speed over the ground less the wind.

## Aircraft by ECAC Doc 29

`acoustic_spaces.doc29` gives the noise of one movement of an aircraft, a departure or
an arrival, at receivers on the ground by ECAC Doc 29, the European method for noise
contours around civil airports (*Report on Standard Method of Computing Noise Contours
around Civil Airports*, 5th edition, 2026). Like the vibration below, it is an
assessment by the method's own empirical curves and corrections, apart from the
scene's sound: nothing of it goes through the simulation.

- **The aircraft** (`doc29.tables`): its NPD curves, the sound exposure level (SEL)
  and the maximum level (LAmax) at 200 to 25,000 ft from a straight, steady flight at
  160 kt for each power, its engines' installation (on the fuselage, under the wings,
  propellers) and its fixed-point profiles, the height, speed and power along the
  ground track from the start of roll or to touchdown.
- **The flight path** (`doc29.flight_path`, Volume 2 section 3.6): the profile along a
  ground track (`doc29.Track`), carried on to the track's ends and the track to the
  profile's, cut into straight segments: the take-off and landing rolls in steps of 10
  m/s, the initial climb and the final approach below 4231 ft at the heights that keep
  the lateral attenuation from changing more than 1.5 dB along a segment, and airborne
  segments whose speed changes by more than 20 kt in steps of 10 m/s; the track's
  nodes joined in, and no source lower than 1 m.
- **Each segment's level** (`doc29.segment_levels`, chapter 4) at each receiver: the
  NPD curves at its power and distance (no nearer than 30 m), corrected for its speed
  against 160 kt, the engines' installation round the roll axis, the lateral
  attenuation of sound passing low over the ground, its finite length, behind a
  take-off roll the start-of-roll directivity, and the air's impedance at the
  aerodrome against the NPD curves' (`doc29.impedance_db`).
- **The event** (`doc29.event`): the segments' energy summed to the SEL, their greatest
  level the LAmax.

`doc29.volume3` holds the reference cases of Doc 29 Volume 3: three hypothetical
aircraft, which rely on no data of the ANP database (a jet with its engines on the rear
fuselage, `JETF`; one with them under the wings, `JETW`; a turboprop, `PROP`), each
arriving and departing on a straight and a curved route at a hypothetical aerodrome,
and the levels Volume 3 gives at its 18 receptors:

```python
import numpy as np
from acoustic_spaces.doc29 import volume3

case = volume3.Case.named("JETFDC")  # JETF departing on the curved route
levels = case.event(np.array([[6500.0, 0.0, 0.0], [8200.0, -1800.0, 0.0]]))
levels.sel, levels.lamax  # dB(A) at each receiver (x, y, z in m)
```

An arrival's profile measures its distances to touchdown and its track from the
landing threshold: `flight_path(..., threshold_m=...)` takes the profile's distance at
the threshold. Volume 3's reference cases put it at the approach's lowest node above
the runway (`doc29.path.lowest_approach_node`), the jets' profile point at 15.2 m, 290.2
m before touchdown, and the turboprop's cut at 17.2 m, 328.1 m before it.

**Flight profiles from procedures** (`doc29.performance`, Volume 2 Appendix B). An
aircraft's fixed-point profile can also be computed from the steps of its procedure
(the take-off; climbs at a constant calibrated airspeed; accelerations at a rate of
climb or with a share of the thrust; level flight; on approach the descents, level
and idle steps, the landing and the decelerations on the runway), its engine and
aerodynamic coefficients and its weight, at the aerodrome's elevation, temperature
and sea-level pressure and into its headwind (`performance.Aerodrome`; by default sea
level in the standard atmosphere into 8 kt, the conditions the coefficients are given
for). Where the thrust changes from one step to the next, a point 1000 ft on (or half
the step, if that is shorter) takes the change, as Appendix B does. The calculation
is in feet, knots and pounds, as Appendix B's is; `performance.as_profile` gives the
points in metres for `flight_path`:

```python
from acoustic_spaces.doc29 import performance, volume3

plane = volume3.performance()["JETW"]
steps = volume3.departure_procedures()[("JETW", "ICAO_A")]
hot = performance.Aerodrome(elevation_ft=5000.0, temperature_c=40.0, headwind_kt=0.0)
points = performance.departure(plane, steps, plane.weights[("D", "1")], hot)
profile = performance.as_profile(points)  # for doc29.flight_path
```

**The ANP database**, the method's data for real aircraft (EASA's Aircraft Noise and
Performance database), is not part of the package: its terms of use (issue 3, 2024)
allow the data only for noise contours of an average fleet and traffic, not for
benchmarking aircraft types, and forbid publishing it or passing it on to anyone who
has not accepted them. With a copy downloaded from EASA under those terms,
`doc29.anp.read_database(folder)` reads its tables from the folder by their files'
names (aircraft, NPD, spectral classes, fixed-point profiles, jet and propeller
engine coefficients, aerodynamic coefficients, default weights, departure and
approach procedural steps), and `Database.profile` flies an aircraft's procedure where
the database has it, else its fixed-point profile. The readers take the fields
Volume 2 Appendix G lists, as Volume 3's workbooks lay them out; they have not been
tried on a download of the database.

Its levels agree with all of Volume 3's reference results: the 99 levels at its
receptors within 0.006 dB, its 12 grids of 37,458 points at 0.003 dB rms, and its
3516 rows of segment results term by term, the segments' places within 7.3 m: the
differences come from where Volume 3 puts its cuts by speed and from its conversion of
metres to feet ([the check](validation.md#aircraft-noise-against-doc-29-volume-3)).
Its flight profiles agree with all 26 of Volume 3's (Part 2: JETW and PROP arriving
and departing at sea level and at 2000 and 5000 ft, at 15 and 40 °C, into headwinds of
-5 to 8 kt): their 314 points' heights, speeds and thrusts to Volume 3's rounding,
their distances within 0.15 ft. Volume 3's profiles differ from Volume 2's printed
equations in three places, where the package follows Volume 3
([the check](validation.md#flight-profiles-against-doc-29-volume-3)).

**The NPD curves in the aerodrome's air** (`doc29.atmosphere`, Volume 2 Appendix D).
The NPD curves hold the attenuation of the AIR-1845 atmosphere, an average of no one
temperature and humidity. `atmosphere.adjusted(curves, spectral_class, temperature_c,
humidity_percent, pressure_kpa)` adjusts them to another air by Appendix D's three
steps: the spectral class taken back to the aircraft, out to each NPD distance through
both atmospheres, and the difference of their A-weighted levels added to the curves.
The aerodrome's absorption is ISO 9613-1's, averaged over each third octave. It gives
Volume 2's worked example (JETW's spectral classes at 10 °C, 80 % relative humidity
and 101.325 kPa, its tables D-3c and D-4) within 0.1 dB, the tables' rounding.

**An aircraft in a scene** (`doc29.source`). Doc 29 gives an aircraft's sound as levels
at receivers, so a source of a scene that sends the same sound is made from them:

- *its sound power* from its spectral class (the spectrum 1000 ft under it at the time
  of its maximum level), scaled to its NPD curves' maximum level 1000 ft away at its
  power, with the AIR-1845 attenuation over those 1000 ft added back and the reflection
  off the soft ground the NPD curves are measured over (1.2 m above it) taken out:
  grass, its reflection's energy added to the direct sound's (`sound_power_db`,
  `ground_gain_db`);
- *its directivity* from its engines' installation correction (eq. 4-15), by the
  depression angle across the flight path, the same in every band; nought straight
  down, where the NPD curves are measured, so the sound power is raised to keep the
  level there (`directivity`, `below_offset_db`);
- *its thrust along the flight*: its sound power changes with the thrust as the NPD
  curves' maximum level 1000 ft away does, its spectrum kept (`power_change_db`);
- *its flight* as a walk's keyframes (`flight`): the flight path's segments flown as
  Doc 29 interpolates them (along a segment the square of the speed changes evenly with
  the distance), so each place is reached when the aircraft gets there, the aircraft
  facing the way it flies.

The airport example's [flight demo](demos.md#aircraft-taking-off-and-landing) flies
Volume 3's JETW this way. The NPD curves are measured at 160 kt and give no
directivity along the flight path but behind a take-off roll (eq. 4-22): the source
keeps none of that either, and the engine reckons the ground, the air and the
lateral spreading in its own way, so its levels differ from Doc 29's; how far is
[measured](validation.md#an-aircraft-in-the-engine-against-doc-29).

What it leaves out for now: bank angles from the turns (a segment takes the bank it is
given; Volume 3's cases fly level, and the performance calculation flies turns as
straight), the dispersion of tracks round a backbone, the line-of-sight blockage of
terrain, levels summed over many movements (LAeq, Lden) and contours. Volume 3 gives
no reference maximum levels, so the LAmax is not checked, and no idle steps of an
approach away from sea level in the standard atmosphere, which Appendix B adjusts so
that the aircraft slows as it does there: those are checked against that definition
only.

## Vibration through the ground

Trains, trams and construction plant shake the ground as well as the air: a building
near them feels it, and its rooms rumble with the sound their floors, walls and
ceilings radiate (ground-borne noise). `acoustic_spaces.vibration` gives both by the
general method of the US Federal Transit Administration (FTA, *Transit Noise and
Vibration Impact Assessment Manual*, FTA Report No. 0123, 2018). It is an assessment
from the manual's curves and tables, apart from the scene's sound: nothing of it goes
through the simulation.

- **Trains and trams** (the manual's General Vibration Assessment, section 6.4). A base
  curve gives the overall rms vibration level of the ground 10 to 300 ft (3.0 to 91.4
  m) from the track's centreline, as far as the manual draws it (figure 6-4, the
  equations of table 6-10): one for trains hauled by locomotives, one for rapid transit
  and light rail (multiple units and trams too), one for a bus on a smooth road
  (`vibration.CURVES`; each train of the railway demo has its own, `rail.Train.curve`).
  The level changes with the speed as 20 log10 of it (eq. 6-4) and, in `vibration.Line`,
  with the vehicles' suspension and wheels, the track's condition and special trackwork
  near, a floating slab, ballast mats, resilient fasteners or ties, an elevated
  structure or a subway's, and the ground: +10 dB for soil that propagates efficiently,
  +2 to +9 dB for a rock layer (tables 6-11 and 6-12, with the manual's rules for which
  add up). `vibration.indoor_db` takes it into a building and up to a floor: the
  foundation's coupling loss, 2 dB a floor up to the fifth and 1 dB a floor up to the
  tenth, and 6 dB for the floors' resonances (tables 6-12 and 6-13). Where a building's
  structure is not known, the manual's conservative change from outdoors to indoors
  stands for all three: +3 dB for a small masonry building or a light wood-frame
  house's first floor, +6 dB for its second and third, 0 dB for a large building.
  `vibration.noise_level` gives the rumble in dB(A): the room's vibration level less 50,
  35 or 20 dB as the vibration peaks below 30 Hz (most surface track), at 30 to 60 Hz
  (most subways, surface track on very stiff clayey soil) or above 60 Hz (subways in
  rock) (table 6-14).
- **Construction plant** (section 7.2). Table 7-4's 14 kinds of plant, from an impact
  pile driver to a small bulldozer: the peak particle velocity (PPV) and rms level 25 ft
  away, falling as (25 ft / D)^1.5 and 30 dB a decade (`vibration.plant_ppv`,
  `vibration.plant_level`, eq. 7-2 and 7-3), against the PPV that may damage four kinds
  of building (`vibration.DAMAGE`, table 7-5), from 12.7 mm/s for reinforced concrete,
  steel or timber to 3.05 mm/s for buildings extremely susceptible to it. The street's
  breaker is the table's hoe ram, its dumper its loaded trucks
  (`plant.Machine.vibration`).
- **Limits** (tables 6-2 to 6-4, `vibration.criterion`): the indoor vibration and
  rumble above which there is impact, for buildings with equipment sensitive to
  vibration, for residences and for institutions used mostly by day, and for concert
  halls, studios, auditoriums and theatres, by how many events a day: frequent (more
  than 70), occasional (30 to 70) or infrequent (fewer than 30). For residences with
  frequent events they are 72 VdB and 35 dB(A).

Levels are in VdB, decibels re 1 micro-inch per second (25.4 nm/s), as the manual gives
them, and a PPV in mm/s and in/s.

The railway example's `vibration.py` stands a house where each of the scene's listeners
in use is (or those `--listeners` names, in that order), as far from the track, and prints for each of the demo's trains at its own speed
the vibration of the ground outside, of the house's floor and the rumble in its rooms
against the limits for residences, and from how far each limit holds. The house is a
small masonry building whose structure is not known, the trains the scene's one
regional train an hour, 24 a day, which are infrequent events; `--building`, `--floor`,
`--ground`, `--peak` and `--trains-per-day` change them. A freight train's locomotive
counts as an infrequent event and its wagons, which pass for minutes, as frequent ones,
as the manual says; it puts the locomotive "often" 3 to 8 dB over the wagons, and the
wagons are taken 3 dB under it. 25 m from the track:

| train | speed | ground | indoors | rumble | vibration within its limit from | rumble from |
|---|---:|---:|---:|---:|---:|---:|
| S-Bahn | 120 km/h | 72.7 | 75.7 | 25.7 | 14 m | 3.0 m |
| regional | 160 km/h | 86.3 | 89.3 | 39.3 | 63 m | 16 m |
| intercity | 200 km/h | 88.2 | 91.2 | 41.2 | 76 m | 20 m |
| freight, its locomotive | 100 km/h | 82.2 | 85.2 | 35.2 | 43 m | 9.7 m |
| freight, its wagons | 100 km/h | 79.2 | 82.2 | 32.2 | 69 m | 18 m |
| tram | 50 km/h | 65.1 | 68.1 | 18.1 | 3.5 m | 3.0 m |

(VdB, the rumble in dB(A); the limits are 80 VdB and 43 dB(A), the wagons' 72 VdB and
35 dB(A); 3.0 m is where the curves start.) The street's `vibration.py` does the same
for the tram, 6.0 m from the houses, its passes frequent events as the manual has most
rapid transit's: 77.6 VdB indoors, 5.6 dB over the limit, and the rumble 27.6 dB(A),
7.4 dB under it. Its breaker, 4.0 m from the houses, gives them a PPV of 5.94 mm/s,
over the damage criteria of the two most fragile kinds of building, and 95.4 VdB, 23.4
dB over the limit for residences; the dumper gives 5.08 mm/s, over the most fragile
kind's and 0.1 % under the next one's (0.1998 against 0.2 in/s), and 94.4 VdB. The
office block, 8.0 m from both, stays under every damage criterion; the method gives it
no limits for annoyance, as it counts offices only where something in them is
sensitive to vibration. Both scripts run from the editor's **Run demo** window. The
pass-by demos play the rumble at each listener, with the train or the tram through a
closed window (`--indoors`, see [Trains passing by](demos.md#trains-passing-by)).

What the method leaves out: it gives overall levels, not their spectra, so the rooms
are not simulated. Its curves are the upper range of measurements of North American
systems in good condition, which the manual says ground-borne vibration rarely exceeds
by more than 1 or 2 dB unless wheels or rails are worn; European trains and soils may
differ. Its table of speeds runs from 20 to 60 mph (32 to 97 km/h); the faster trains
take its 20 log10 of the speed further. The manual's detailed method works band by
band from a train's force density and a site's transfer mobility, which it says to
measure (see the [roadmap](roadmap.md)).

## In the editor

The editor builds and edits outdoor scenes too: see
[Outdoor scenes](editor.md#outdoor-scenes).

## What is not modelled yet

The wave solver's moving air (`simulation.wave.wind: moving`) runs on the CPU only,
and leaves out the term of the linearised Euler equations that the wind's change with
height puts on the air's own motion, about |dv/dz| / ω of the rest (under 0.3 % at 63
Hz a metre above the ground in the `windy` preset). Its wind is the weather's profile
everywhere, as for the rays: no flow is computed round buildings, so the wind blows into
a façade as over open ground and a building's lee is as windy as the open. By default
it takes the wind as
the rays and the parabolic equation do, as an effective speed of sound, so sound that
a building sends back towards the source travels at the speed of sound going away from
it. It leaves out vegetation and the eddies.

Vegetation is volumes filled evenly: a lone tree's trunk is left out, the foliage's
attenuation is given, not derived from its leaves' size and density, and stems stand
upright at random.

Sound bent over the top of a barrier or a roof keeps straight legs in any weather, its
level falling only past the start of a refraction shadow: the wind and the temperature
do not change how much a barrier screens, though downwind and under an inversion it
screens less in practice (ISO 9613-2 and CNOSSOS-EU take that as a path curved over
the edge). Behind a barrier, then, the weather classes of a [long-term
level](#levels-over-a-long-term) differ only by the sound that does not pass over its
edge, the late sound above all.

The eddies are the same over the whole scene (weather zones change the wind and the
temperature, not the eddies), and the eddies' largest size is the same at every
height. A weather zone's border is a step, and the early paths take one mean profile
along their way, so a wind that changes on the way tells on them by a few decibels in
the bands the rays give (see [Weather zones](#weather-zones)). The ground reacts locally unless
`reaction: extended` says otherwise, and the wave solver's ground always does: one
impedance for every angle, which holds for grass, soil and snow, less for very soft
snow and thin layers.

Over a terrain the early sound reflects off the mean plane of the ground's profile, not
off its facets, so a listener high over rolling ground gets one reflection where the
real ground glistens with many, and a valley between the source and the listener
reflects as its mean plane does, not as a bowl. A rounded crest is a wedge of its
chords with a thick screen's extra screening, an engineering model as in the standards,
where a ridge is exact. Sound bent over a hill and a building in turn is left out, and
a terrain cannot be combined with wind or temperature gradients yet. The editor shows a
terrain and the buildings of a map but does not draw either: they are set in the scene
file.

Sound that reflects off the ground both before and after bending over an edge (two
reflections and a diffraction) is left out unless
`simulation.early.diffraction_reflections` is `both_sides`: by default a path
reflects before or after the edge, not both. Over a rigid ground that path carries a
quarter of the sound over a barrier, so without it the barrier's insertion loss comes
out up to 0.8 dB off from 1 kHz up (mostly too high, 0.7 dB too low at 2 kHz 90 m
behind) and a few decibels off in the bands below, where the paths interfere; with it
the engine is within 0.15 dB of the exact solution (see
[Validation](validation.md#outdoor-results-against-exact-solutions)). Over grass and
other soft grounds, whose reflections near grazing are weaker, the path left out
should matter less, but that is not checked.
Sound that goes over a barrier, down to the ground and over the barrier again is left
out either way.
