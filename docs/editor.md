# Graphical editor

The editor needs the `gui` extra: `uv sync --extra gui` in a source checkout, then
`uv run acoustic-spaces`, or a release wheel installed with `[gui]` after its file
name.

```sh
acoustic-spaces                                   # alone, it opens the editor on a new box room
acoustic-spaces edit examples/studio/scene.yaml   # or open a scene file
```

`acoustic-spaces` with no command opens the editor; `acoustic-spaces -h` lists the
commands. Without the `gui` extra, or with no display to show the editor on (over SSH,
say), it prints the short usage instead, with the reason.

![The editor with the concert hall example: the scene tree, the room with its early reflection paths, the source's properties and the echogram of a run](images/editor.png)

The editor works on the same scene files as the command line. Rest the mouse on any
label, field, button or column header for a tip that says what it means, with its unit:
the rows of the Results table say what each parameter measures, and the band columns
which frequencies each octave band holds.

- **File > Open example** opens the bundled rooms, in the groups of
  [Example scenes](examples.md). Saving one asks where to keep your copy and takes the
  files it uses along (its model, terrain, tables, directivities, and the patterns its
  array files name); an example whose model needs an extra that is not installed is
  greyed out, and its tip says how to install it. **New box room**, **New
  room from a floor plan**, **New from model file** (OBJ, STL, PLY, OFF, glTF, DXF,
  IFC, Collada or SketchUp) and **New outdoor scene** (a meadow 200 m across, to build
  on; see [Outdoor scenes](#outdoor-scenes)) start from scratch. For an IFC model, the
  geometry's form picks one room of the building (**IFC room**) or every element.
- The **Plan** tab, next to the 3D view, shows the room from above and draws rooms (see
  below).
- The **scene tree** lists the geometry and its surfaces, the materials, sources,
  listeners, the wind outdoors, simulation settings and output formats. Selecting an entry, there or by
  clicking it in the 3D view, shows its form under **Properties**. Every change is
  checked against the scene schema; a refused change says why under **Messages**, and
  every change can be undone. **Properties** takes the window's whole height on the
  right and is never narrower than its widest form needs (up to half the screen), the
  window opening wide enough for it; a row too long for it puts its field under its
  label.
- Sources and listeners sit in their **groups** (the form's **Group** field; groups
  nest with `/`, as PA/Delays). Each source, listener and group has a tick: unticked,
  it is left out of simulations, maps and checks, stays in the scene drawn faded, and
  is saved as `enabled: false` (the form's **Use in simulations** does the same). The
  eye beside it hides it in the 3D view and the plan, or shows it again, without
  changing the scene. See
  [Groups, and sources left out](scenes.md#groups-and-sources-left-out).
- The markers of sources and listeners in the 3D view take their size from the
  scene: a fifth of the usual spacing between them, from 1.2 % to 2.5 % of the model's
  size. Where they crowd, such as the players and monitors on a stadium's stage, they
  get smaller rather than overlap, in the plan too, and an arrow stops short of a
  marker ahead of it; hiding a group gives the rest their size back.
- **Place source** and **Place listener**, then click a surface: on a floor the new
  point stands at ear or mouth height, on a wall or ceiling it goes half a metre into
  the room. In the plan, the click places it above the floor there. Sources and
  listeners move by dragging, in the 3D view or the plan, and keep their height above
  the floor (up a raked floor too); with Shift in the 3D view they move up and down.
  Surfaces take materials from their form, or from a new material picked from the
  database (**Edit > Add material from the database**) or from a CATT-Acoustic or ODEON
  library (**Edit > Add material from a library**).
- A material's form sets its absorption and scattering per band (24 columns when it
  has third-octave data), whether it **Lets through** nothing, part of the sound (with
  its sound reduction index per band in the table) or everything (an opening), and
  **Audience seating** for the seats (see
  [Walls, doors and openings](scenes.md#walls-doors-and-openings)).
- **Edit > Edit movement…**, or **Edit movement…** in a source's or listener's form,
  opens the **Movement** panel: the walk of the selected point source or listener (see
  [Walks](scenes.md#walks)) as a table of keyframes, where it is and how it faces at each
  time, with the row 0 where it starts. No speed is set: **Add keyframe** puts one at the
  playhead, where the mover would be then (on a keyframe already, or at 0 s, a second
  past the end of its walk), to drag into place in the plan or the 3D view (its handles
  are numbered as the rows; Shift moves one up and down in 3D) or to type where it is;
  the **Speed** column shows what follows, the distance to the next keyframe over the
  time to it. A keyframe at the same place waits there, one with other angles turns on
  the spot; dimmed angles keep facing as at the keyframe before, and typing one turns
  it there. Every change is one step of the history, as the forms' are. The timeline
  below runs to the longest walk at least; make **End** longer to add keyframes past
  it. The slider (or **Play**) moves every mover in the 3D view and the plan to where
  it is then, nothing simulated; **Moves** (in the panel and the forms) keeps a walk but
  leaves its mover standing, as `enabled: false` does, and **Doppler** has it play as a
  vehicle does, each early path through a delay that changes as it goes, so its pitch
  shifts as it passes (`doppler: true`; a walk with Doppler moves alone, so the other
  walks must be off). A line or a surface source does not move: its form shows
  **Moves** and **Edit movement…** greyed, with a tip saying so. A walk through points at
  one speed (`to`, as the car of the `roadside` and `street` examples has) is edited the
  other way round: a row per point, its **x**, **y** and **z** typed or its handle
  dragged, the walk's one speed typed in the **Speed** column (m/s; the hint gives it in
  km/h too), and the times following from them, read-only, as do the angles of a walker
  that faces the way it goes. **Add point** splits the leg the playhead is on where the
  mover is then (at 0 s or on a point, it adds one a second further on past the end,
  along the last leg), and **Remove point** takes the selected one away; **Convert to
  keyframes** bakes the walk into keyframes to time each place, its turns through the
  corners included (facing forward, it faces its first leg from the start, so its own
  orientation turns that way). The columns give their units. A dragged keyframe or point
  keeps its height above the floor, as a dragged source or listener does. **Render
  movement…** simulates every mover on one clock and writes what each listener in use
  hears of what the sources play (their **Plays** files) as a WAV file, in the
  background, as `walkthrough --all` does (see
  [Walkthroughs](spatial-audio.md#walkthroughs)), in the output **Output** picks: the
  scene's outputs, binaural ones first, and binaural through a spherical head model first
  when the scene has none (the `roadside` lists mono only), since a source passing by
  goes from one ear to the other only in binaural. A walk with Doppler plays there too
  when it is the only one that moves. One gain brings the loudest peak to -1 dBFS, so
  the balance between the listeners holds; the messages name the output.
- The scene tree's **Simulation** entry has the settings of
  [Simulation](simulation.md): the early part's method, highest order, rays,
  diffraction, **Reflections over an edge** (on one side of it or on both) and **In
  pressure up to**; the late part's method, rays per band,
  reception, arrival directions, **Batches** (from 2, standard errors for the
  parameters), ray splitting and **Run on** (*default*: this computer's default, the CPU
  unless the tune window made it *auto*; or the CPU, a GPU, both, or *auto*); the
  scattering model; the wave solver, its highest band, how it takes the wind outdoors
  (**Wind**: an effective speed of sound or moving air) and where it runs; **Performance** (CPU
  threads and the ray search tree); and, under **Response**, the length, the sample
  rate, octave or third-octave **Bands**, the early-to-late transition and the seed.
  The scene's own entry sets the air's temperature, humidity and pressure, and shows
  the volume and the rooms the model holds.
- **Simulation > Tune CPU and GPU…** times the CPU and the GPU on the open scene (or
  on the concert hall example when the scene cannot run as it is), as
  `acoustic-spaces tune` does (see
  [Tuning the devices for a computer](simulation.md#tuning-the-devices-for-a-computer);
  needs the `gpu` extra). It runs in the background, once per computer, and then shows
  the table of times with a check box, **Make auto the default on this computer**
  (ticked): scenes whose **Run on** is *default* then run on the fastest device found,
  in the editor and on the command line. When the open scene names its own devices, a
  second check box (ticked) sets its **Run on** back to *default* (an edit like any
  other, undone with Undo). **OK** saves the timings and these choices; **Cancel**
  saves nothing, not even the timings.
- **Simulation > Run demo…** runs a [demo](demos.md) of the open scene. *This scene*
  plays the WAV file each source's **Plays** names through it, mixes every listener and
  makes a video ([a demo of any scene](demos.md#a-demo-of-any-scene)); the scene of an
  example that comes with a demo (the orchestra in the concert hall, the choir and organ
  in the church, the band in the stadium, the concert across the lake, the walk through
  the gallery, the car passing the roadside and driving down the street), or a copy of
  one, also has that demo, reading the open scene, and the window starts on it (a copy
  is known by its folder's name or, saved anywhere, by the scene's name). Pick
  the demo and the step (*all* runs it from nothing to its video, leaving out what is
  done already), fill in the options, which start at the same defaults the command
  line uses (the **--input** table lists the sources, what
  each plays and the WAV file it plays instead: **Choose WAV…** gives the selected one a
  file, **Default** takes it back; the **Movement** box ticks the sources and listeners
  whose walk moves in the run, which *This scene* lets you change for that run,
  `--moving`, a walker left out of the scene showing only when the demo moves it, and
  **Edit movement…** opens the Movement panel on the selected one; a number shows its
  unit, and the pass-by demos' **--speed** reads *as in the scene* at zero, the speed of
  the car's walk, with the line under the Movement box saying how far, how fast (km/h
  and m/s) and so for how long the car drives, as the options stand), and
  it runs in its own process in the background, with what it prints in the **Messages**
  panel and **Cancel** to stop it.
  The window shows the command line it is about to run, so the same run can be
  repeated in a terminal, and asks for a scene with changes to be saved first, as the
  demo reads the file. An example's demo plays recordings it downloads, or a sound it
  makes (the pass-by demos' car), and can take a while. A demo writes beside the scene
  (*This scene* into `demo/`), or under your home folder when that folder cannot be
  written.
- The scene tree's **Outputs** entry chooses what a run writes: mono, binaural,
  Ambisonics or loudspeakers. For binaural, **HRTF file…** takes a SOFA file from the
  disk, **Spherical head** goes back to the model, and **Download HRTFs…** offers the
  free sets with their licences and fetches the one chosen in the background (see
  [HRTF sets to download](spatial-audio.md#hrtf-sets-to-download)).
- **Simulation > Run coverage map** (F6) maps every source over the audience area (see
  [Coverage maps](coverage-maps.md#coverage-maps)). The **Coverage map** panel shows any parameter
  and band from above, for one source or all of them together, and **Map in 3D** lays
  it over the room, see-through so that the sources and listeners stay in sight;
  **Save** writes the picture or every map as JSON. The scene tree's
  **Coverage map** entry sets the surfaces, height, spacing, rays and background noise.

  ![The editor with the lecture hall example and its speech transmission index map laid over the room](images/editor_map.png)

- **Simulation > Run** (F5) works in the background. The **Results** panel shows the
  ISO 3382-1 parameters (outdoors also the level relative to free field, the insertion
  loss and the level over a long term, see [Results outdoors](outdoor.md#results-outdoors)
  and [Levels over a long term](outdoor.md#levels-over-a-long-term)), the echogram
  and decay curves and the arrival directions of any source and listener, and draws
  their reflection paths in the room. With the late rays in batches, each parameter's
  tip gives its standard error, how much it would change between runs with other seeds
  ([Simulation](simulation.md)), and the value is amber where that is over half a JND:
  more rays per band would make it smaller. **Export**
  writes the same files as `simulate`, in the background (**Cancel** stops it);
  **Listen** plays a dry WAV file through the
  response on the sound card, binaurally (through the HRTF of the scene's binaural
  output if it has one, else a spherical head; needs the `measure` extra), or
  head-tracked with **Head tracking** ticked (see
  [Head tracking](spatial-audio.md#head-tracking)). While it plays, the status bar says
  so, with **Stop playback** (also in the toolbar and the Simulation menu, or Esc).
  Picking another source or listener (in the scene or the panel), or ticking **Head
  tracking**, plays on from the same point with that response, at the same level so
  the difference stays audible. What plays does not follow changes to the scene (moving
  a source, say): the status bar marks it as from before them, and Listen again plays
  them.
- Change a material's absorption after a run and the results follow at once: the run
  counted each ray's reflections off each material, and is re-weighted for the new
  values instead of traced again (as `calibrate` does). Where the early reflections add
  in pressure (up to 125 Hz indoors by default, every band outdoors), their phases turn
  with the absorption too, as a new run has them; only the faces of an edge that sound
  bends round keep what they did to it as traced. The panel says so; **Run** traces
  afresh. Other changes, and runs with the statistical method, third-octave
  bands or the wave solver, leave the results marked out of date until the next run.
- A source's form sets its **Shape** (a point, a line or a surface, with its length,
  height and spacing), its directivity (a pattern, a SOFA or CSV file of measured data,
  or an array file, whose elements **Add up the array's elements one by one** sums near
  its listeners), its sound power per band, its **Spectrum** (flat, or male or female
  speech) and the **Gain**, **Equalizer** and **Delay** of its feed. It shows the
  directivity index per band, opens the polar patterns, and the selected source shows
  its balloon in the 3D view. **Plays** names the WAV file the source plays in a demo of
  the scene: **Choose WAV…** picks a dry recording, **Silent** takes it away.
- The status bar keeps the volume and the Eyring reverberation time at 1 kHz up to date
  as you edit, with the number of problems in red.
- **Messages** shows warnings (leaks, results from before the latest changes, dropouts
  while playing) in amber and errors (a refused change, a run that cannot start or
  failed, a file that cannot be read or written) in red; so does the scene's own form
  for its problems and warnings. Every other message is shaded, so that one ends where
  the next begins. The ticks above them show or hide the info, the warnings and the
  errors (with how many there are of each); **Item** keeps only the messages that name
  one source, listener or material (the list holds those that messages name, with how
  many do), and the filter keeps only those holding some words. **Clear** removes the
  messages on show and keeps those hidden.
- While a run, a coverage map, a tuning or an export goes on, the status bar says what
  runs, what it does now and for how long, beside a moving bar and a **Cancel**
  button; resting the mouse on the bar says so at length.
- When a run, a coverage map or a tuning ends while another window is in front, the
  editor's entry in the taskbar flashes (on macOS, its Dock icon bounces). Nothing pops
  up: the results come to the front and **Messages** says how it went.
- **View > Show leaks** traces rays from the sources and shows where they escape from a
  model that is not closed: its open edges, the escaping rays and a marker on each gap,
  with the report in the messages (see [Finding leaks](scenes.md#finding-leaks)). It
  traces again when the geometry, the materials or the sources change.
- **View > Show wind** draws the wind of an outdoor scene in the 3D view and the plan:
  sky blue arrows the way it blows, as long as it is strong, at the height the scene
  gives its speed at, each weather zone with its name and wind, and the scene's wind in
  a corner. **View > Show wind profile** stands a mast in the 3D view in each weather
  where the wind blows, with arrows up it and a curve through the wind at every height:
  how the wind grows with height over the ground (see
  [Wind and temperature](outdoor.md#wind-and-temperature)). Both are on at first. The
  scene tree's **Wind** entry, outdoors with wind or temperature, has an eye for each
  (its own, and that of **Wind profile** under it) and lists the weather zones with
  their winds; selecting one shows its weather in the scene's form.
- **View > Language** shows the editor and the command line in English, Spanish,
  Catalan, Italian, German or Dutch, or in the computer's language, from their next
  start (see [Language](getting-started.md#language)). Numbers in the editor follow the
  computer's regional settings: 0,85 where the decimal mark is a comma, with lists of
  numbers (an equalizer's gains) apart by semicolons there.
- **File > Pack for another computer** writes the scene and every file it names to one
  zip file, as `acoustic-spaces pack` does, to run on a server or a cloud job (see
  [Running on another computer](remote.md)).
- **File > Export to Blender** writes the room with the reflection paths on show, and
  **File > Export particle animation to Blender** the sound particles leaving the
  selected source, asking first how much of the sound to animate, how slowly and how
  many particles (see below).

## Drawing rooms

![The plan of the gallery example while a corner is dragged, with the lengths of the two walls it moves](images/plan.png)

The **Plan** tab shows the room from above on a grid in metres. In a floor plan or box
room every corner is a handle:

- Drag a corner to move it; the two walls it joins show their lengths as it moves. A
  box becomes a floor plan with the same walls, so its materials stay where they were.
- Double-click a wall to add a corner there, or right-click a corner to remove it.
  Clicking a wall selects that surface, to give it a material.
- **Draw room** (or **Edit > Draw floor plan**) draws a new floor plan: click its corners
  in turn and close it on the first one, with a double-click or with Enter. Backspace
  takes the last corner back and Esc stops. The new room keeps the old one's height
  (change it under **Geometry**). Sources and listeners that would end up outside move
  inside, keeping their arrangement. Over a model file, the editor asks before
  replacing it.
- Corners and points snap to the grid chosen under **Snap** (10 cm unless changed);
  hold Shift to place them freely. A move that would make walls cross is refused, with
  the reason under **Messages**.
- The wheel zooms, dragging empty space pans and **Fit** shows the whole room again.

Rooms from model files show their outline, and their sources and listeners move the
same way. Every change can be undone.

## Outdoor scenes

**File > New outdoor scene** starts an [outdoor scene](outdoor.md): a flat meadow 200 m
across under an open sky, with a source and a listener 50 m apart. The **Geometry**
entry of an outdoor scene has an **Outdoors** group instead of the room's: the
ground's height, the margin, the sky and the **Extent** (tick it to set where the
ground ends; unticked, the ground reaches the margin past what stands on it), then the
buildings, barriers, [vegetation](outdoor.md#vegetation) and ground zones in tables,
with their heights, which can be changed, and buttons to add and remove them:

- **Add building**, **Add zone** and **Add vegetation** with *belt of trees* chosen
  draw an outline in the plan, like a floor plan: click its corners in turn and close
  it on the first one, with a double-click or Enter. **Add barrier** and **Add
  vegetation** with *hedge* chosen draw a path: click its points and finish with a
  double-click or Enter. A *lone tree* stands where the next click lands. Backspace
  takes a point back and Esc stops. A new building is 10 m high, a barrier 3 m, a belt
  of trees 12 m, a hedge 2 m high and 1.5 m wide, a tree 10 m high with a crown 8 m
  across; change the heights in the tables and the rest in the scene file. A zone takes
  the ground's material until its surface is given one in the scene tree.
- **Terrain from a file** makes the ground follow the heights in an Esri ASCII grid or
  a table of x y z points (see [Terrain](outdoor.md#terrain)); the form then describes
  it (its points, heights and grid), and **Remove terrain** flattens the ground again.
- **Buildings from a map** adds the buildings of a GeoJSON file of footprints (see
  [Buildings from a map](outdoor.md#buildings-from-a-map)), after asking which of its
  properties holds their heights (or one height for all of them), the height of those
  without it, and which property names them. They stand in the buildings table after
  the scene's own, where **Remove building** leaves one out of the file, and **Remove
  the map's buildings** drops the file.

The plan draws the vegetation in dashed green and the zones in their materials'
colours; the 3D view draws the vegetation in see-through green and leaves the sky out
but for its edges, which show where the ground ends. The scene's own entry then has a
**Wind and temperature** group: the weather, the wind and the temperature over the
ground, their **Eddies** with their strength, size, **Eddies' height** (the height the
strength holds at; 0: the same at every height) and **Weaker with height** (see
[Turbulence](outdoor.md#turbulence)), and **Show the
profile**, a plot of them and of the speed of sound (see
[Wind and temperature](outdoor.md#wind-and-temperature)). **Weather of** chooses whose
weather the fields show and change, the whole scene's or a
[weather zone](outdoor.md#weather-zones)'s: **Add weather zone** draws one in the plan
like a zone, dotted in sky blue with its name and wind, and **Remove weather zone**
drops the chosen one. A zone's weather starts as the scene's, its fields are those it changes,
(none) under **Weather** leaves the scene's weather unchanged there, and **Ground
temperature** gives the ground in the zone a temperature of its own. **Add weather
class** adds a [weather class](outdoor.md#levels-over-a-long-term), a weather that
holds for a share of the time, for the levels over a long term and L_den: **Weather
of** then shows it, with its **Share of the time** in each period of the day (the
other classes make room, in proportion, when one changes), its ground temperature and,
under **Eddies**, eddies of its own if it has them. The first class holds all the time
in the scene's weather, the next ones none of it until given a share, and **Remove
weather class** gives the chosen one's share to the others. A run then runs the scene
in every class, and the Results panel adds the row **Long-term level** and, under
**Run details**, the rating level and the level in each period and each class. **Draw
room** replaces an outdoor scene with a floor plan, after asking.

## Exporting to Blender

`acoustic-spaces export` writes a scene as glTF 2.0 (`.glb`) for Blender or another 3D
tool; the editor does the same from its File menu.

```sh
acoustic-spaces export examples/concert_hall/scene.yaml hall.glb              # the room
acoustic-spaces export examples/concert_hall/scene.yaml hall.glb --paths      # + reflection paths
acoustic-spaces export examples/concert_hall/scene.yaml hall.glb --particles  # + particle animation
```

In Blender, use **File > Import > glTF 2.0 (.glb/.gltf)** and pick the file. The axes are
the scene's (+z up, metres).

- Each surface is an object named as in the editor, with a see-through material per
  scene material; the material's custom properties hold its absorption and scattering
  per octave band. Sources (red) and listeners (blue) are spheres turned the way they
  face, with an arrow along the aim. Outdoors, each tree, hedge or belt of trees is a
  see-through green object under `vegetation`, and the sky is left out.
- `--paths` adds the early reflection paths from `--source` to `--listener` up to
  `--max-order`, as long after the direct sound as a simulation keeps them
  (`transition_ms`, or the mixing time with the statistical late method, see
  [Simulation](simulation.md)), one object of edges per order. Edges do not render: to
  render them, use Object > Convert > Curve and give the curve a bevel depth.
- `--particles` adds the particles of `animate` as small spheres that move with them and
  shrink with their level, gone 60 dB down. The animation runs 40 times slower than the
  sound (`--slow-motion`): the default 100 ms take 4 s, 96 frames at 24 fps. If the
  timeline ends sooner, set its end frame, then press play. With the default 1000
  particles (`--count`) the file is about 3 MB. The editor asks for the same three:
  **Sound to animate**, **Slow motion** and **Particles**. Outdoors it starts with sound
  enough to reach the farthest listener (a tenth more than the direct sound takes,
  rounded up to 100 ms), slowed down to last about the default's 4 s: 1900 ms at 2.1×
  for the `lakeside_concert` example's village 568 m away. Outdoors most particles soon
  leave through the open sky and fly on straight. Blender draws a dashed line from each
  particle to the object that holds them; **Overlays > Relationship Lines** hides them.
