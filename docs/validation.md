# Validation

This report sums up how far the results of acoustic_spaces can be trusted: what the
test suite checks against theory on every change, how the simulation compares with
another program, [pyroomacoustics](https://github.com/LCAV/pyroomacoustics), and with
measurements in two real rooms, how edge diffraction, reflections fading at edges and
the reflection off a porous ground compare with the exact solution, how sound outdoors compares with measurements
up to 845 m away, how its turbulence compares with a parabolic equation through random
eddies, how its rays through vegetation compare with a separate particle simulation,
how its wave solver in wind and temperature compares with a parabolic equation, and how
much of a block's time the real-time convolver needs. It ends with the known issues and
what has not been validated yet.

## Checks against theory, on every change

The test suite runs on Linux, macOS (Apple silicon and Intel) and Windows for every pull
request and checks:

- an 8 x 6 x 3 m box against analytic image sources up to 4th order (1, 6, 18, 38
  and 66 paths), for both early methods and for symmetric placements where paths
  hit edges and corners: positions, delays and energies agree to rounding;
- T30 in a box with fully diffusing surfaces against Eyring with air absorption:
  within 10 %, and slightly longer as absorption grows, as Kuttruff's correction
  for the spread of free path lengths predicts; also within 10 % with the example's
  uneven absorption (absorbing ceiling and carpet, hard walls) once every surface
  diffuses;
- the decay in the same box with specular walls against its closed form, where each
  direction of travel meets the walls at its own fixed rate: within 10 % with the
  absorption spread evenly, and with the example's, where rays parallel to the floor
  never meet the ceiling and T30 is 5 to 20 times Eyring's estimate;
- G in the same box against Barron's revised theory: within 1 dB;
- the statistical late method against the stochastic one in the same box, with the
  same absorption on every surface and scattering 0.3 and 1: G within 0.7 dB (up to
  0.62 dB at scattering 0.3 and 0.43 dB at 1), C80 within 2.5 dB (higher from 1 kHz up,
  where the rays decay up to 7 % slower than Eyring);
- the late tail of the example box (absorbing ceiling, smooth walls, random
  scattering) from 10k rays with splitting against 1 million rays: within 4 dB on
  average from 0.5 to 2.5 s, and splitting or thinning out every ray keeps the energy
  of plain ray tracing;
- the late part's last slots: the same rays traced 50 ms longer give the same echogram
  up to the end, where a ray's time runs out as it crosses a listener's sphere: with
  and without the rain on the CPU, without it on the GPU, and with rays bent by the
  wind on both;
- scattering: the energy of an early reflection off a wall against Christensen and
  Rindel's scattering of its size, to rounding; the early paths and the late part
  adding up to what the tracer finds on its own within 5 %, with vector-based and with
  random scattering; a box of tiny surfaces decaying at Eyring's rate within 10 % where
  its smooth walls alone decay far slower; and a box that diffuses the bass and
  reflects the treble decaying at Eyring's rate in the bass only, each band traced with
  rays of its own;
- third-octave bands: statistical runs in third octaves give the octave runs'
  parameters within 1 % or 0.1 dB from 250 Hz up (within 5 % or 0.4 dB at 63 and
  125 Hz, where the early paths add in pressure and three filters weigh their
  interference across an octave a little differently from one), and stochastic ones
  within the tracer's noise; a
  100 Hz third-octave absorber decays at its own Eyring rate (within 3 %) between bands
  that do not; the filter banks sum to their input and sit at -3 dB on the band edges;
- sources: the free-field level 4 m from a source with a sound power per band, a gain
  and an equalizer, against 10 log10(4 π r²) and the air absorption of ISO 9613-1 (to
  rounding), with G unchanged by the feed; a delayed source's response 12.5 ms later
  (600 samples at 48 kHz) with the same parameters, its wave bands too; two alike
  sources together 3 dB louder and as clear as one, and less clear with one of them
  150 ms late; the speech spectra of IEC 60268-16 summing to 0 dB(A); and the STI of
  female speech, weighted to the treble, higher where the bass is blurred;
- line and surface sources: a 40 m line's free-field G at 2 and 4 m equal to the sum
  of its 80 points with their air absorption (to 1e-9 dB), falling 3.31 dB per
  doubling of distance as the continuous line does; an 8 x 8 m Lambert surface within
  0.15 dB of the integral over the continuous square at 1 and 2 m, and silent behind;
  in the shoebox a 3 m line as loud as a point source whose paths add in energy, as
  the line's uncorrelated points' do (within 0.6 dB at 2000 rays per
  band) and its T30 within 10 % of the point source's at 30,000 rays per band;
- coherent arrays (their far-field patterns; nearer than L² f / c the room's
  warnings say so unless the array adds up its elements one by one, `near_field`): a
  column of 8 omnidirectional elements 20 cm apart against the
  analytic sums of a line array, its directivity index within 0.05 dB in all 24
  third-octave bands (n² over the band mean of the pairs' sin(kr)/kr) and its vertical
  pattern within 0.3 dB of the band-averaged |sin(Nψ/2)/sin(ψ/2)|² wherever that is
  above -15 dB; a steered column's beam within 1 degree of its aim, and the same as
  delays given element by element; two close elements in opposite polarity a dipole
  (4.77 dB); two cardioids splayed ±30 degrees their exact first-order sum; and in a
  simulation, levels ahead of and above a column that differ by its pattern exactly;
- coherent arrays element by element (`near_field`): a steered column of 8 cardioids
  3 m from its centre against the sum of its elements' pressures over 20,000
  frequencies in each third-octave band, within 0.001 dB in all 24; 100 km away its
  pattern's own weights (within 0.00001 dB); a 3.75 m column of 16 elements on axis
  losing 2.8 dB per doubling of the distance at 1 kHz from 1 to 32 m, inside the 41 m
  where its far field starts (3 dB for a line), and 6.02 dB from 128 m on (a point);
  and in a simulation, levels 4 m ahead of and above a 1.4 m column that differ from
  omni by the elements' sum there exactly, ahead 9.5 dB below its far-field pattern at
  4 kHz, where the beam has not formed yet;
- walls and openings: two 5 x 4 x 3 m rooms, 20 % absorbing, with a 12 m² partition
  between them reach the level behind it that diffuse-field theory gives,
  G = 10 log10(1600 π S τ / A²) with Eyring's absorption area A, within 0.4 dB at R =
  30 and 50 dB (0.1 to 0.2 dB at 3000 rays per band), and their level difference
  R + 10 log10(A / S) within 0.5 dB; two such rooms joined by an opening the size of the
  wall give the T30 (within 4 %), G (0.3 dB) and C80 (0.5 dB) of one 10 x 4 x 3 m room;
  and a dry room of 96 m³ opening through a 2 m² doorway into a live one of 192 m³
  decays in the two slopes of the two-room diffuse model, within 1.5 dB of its decay
  curve down to -30 dB and its late reverberation time within 12 % of the model's slow
  mode (within 1 dB to -40 dB and 5 % at 10,000 rays). Particles reach the next room
  straight through an opening and not through a closed door, and the wave solver's air
  flows through the opening. The statistical method follows the two-room model within
  0.3 dB of its decay curve from -20 to -40 dB in both rooms, and comes within 0.4 dB
  of the rays behind the partitions (0.2 dB below them: its field reaches each point
  from the direct sound on, as Barron has it). Coverage maps over both rooms find at
  their points what listeners there find (G, T30 and C80 within 0.01 with the
  statistical method), and the rays' maps the level behind the partitions within
  0.4 dB;
- audience seating: the grazing attenuation against its fit's own figures (17 dB at
  grazing incidence after 11 m, 6.3 dB at 5° and 4.8 dB at 7°, none from 17.8°,
  linear in dB with distance); the direct sound 18 m over seats at 0.95° 21.1 dB down in
  the 500 Hz to 2 kHz bands only, the ceiling reflection untouched; a balcony seat
  shaded only by its own seats and the front stalls the sound skims, not by the stalls
  far below; with the statistical method, a response that loses exactly what its early
  paths lose (the tail does not make it up); and a coverage map less clear and weaker
  over the middle of the seats;
- fitting materials to measurements: a run re-weighted for other absorption against
  the room traced again with it (the same early paths, with the same energies and,
  where they add in pressure, the same phases to 10⁻¹², those of reflections on the way
  round an edge included; T30 within 4 %, EDT within 5 %, C80 within 0.3 dB and G
  within 0.1 dB); the absorption a simulated room was given
  recovered from its own responses within 0.05 in every band, and a shoebox's walls
  within 0.015 from 500 Hz up, where the fit's own uncertainty is below 0.05 and above
  0.1 in the bands the measurements leave open; a third-octave material that keeps its
  shape; and the two BRAS rooms (see [Measured rooms](#measured-rooms-bras));
- every ISO 3382-1 parameter on synthetic exponential decays, and the air
  absorption against the ISO 9613-1 table;
- the further parameters against responses whose values are known: LF and LFC of one
  reflection from the side, from 45°, from behind and from above, exact from the
  echogram and within 0.01 from W and Y impulse responses; a third of diffuse late
  energy lateral (a half for LFC); LJ as the G of the late lateral energy; the T30 of
  each axis when front-back sound decays in 2 s and the rest in 1 s, exact from the
  echogram and within 5 % from noise; IACC of 1 for equal ears and ears 0.5 ms apart,
  and below 0.15 from 2 kHz for unrelated ones; stage support of known energies to
  rounding; STIPA of exponential decays against Houtgast's closed form within 0.002,
  and within 0.01 of the STI; U50 against its definition; and the echo criteria of an
  echo as loud as the direct sound 100 ms later within 10 % of the jump of the centre
  time, below the 10 % audibility mark for a smooth decay, and for music an echo at 1,
  1.4 or 2 kHz reaching the 50 % mark at the same level within 0.25 dB;
- spatial outputs: SN3D spherical harmonics against closed forms and orthogonality,
  VBAP gains of unit power that point at the panned direction and are mirrored for a
  mirrored direction, the direct sound's direction in first-order Ambisonics, the mono
  energy kept by panned layouts (direct sound within 1 %, late part within 5 %), a
  binaural render through flat HRTFs equal to the mono response, and the spherical
  head's interaural delay (0.6 ms at the side);
  HRTFs interpolated between measured directions give the measured responses at those
  directions and, round the horizon a degree at a time, an interaural delay that moves
  by less than 50 µs per degree where the nearest measured response jumps by more than
  150 µs; mixing pulses delayed by fractions of a sample keeps them flat within 0.5 dB
  up to 16 kHz, and the spherical head's within 1 dB of the nearest measured response
  at 8 kHz; a ring of directions or half a sphere interpolates within itself and
  takes the nearest response outside; the rotation of a sound field of any order turns
  encoded plane waves exactly (to 1e-9), is orthogonal and composes;
- rendering: the compiled FFT against NumPy, the real-time convolver and the offline
  convolution against direct convolution (float32 and float64 precision), resampling
  of tones within 1e-3 of the exact signal up to 15 kHz with nothing above the new
  Nyquist frequency left, and playback through a simulated sound card block by block;
- edge diffraction: the UTD coefficient against Sommerfeld's exact solution for a
  rigid half-plane (to 1e-8, also exactly on the shadow and reflection boundaries) and
  against the exact eigenmode series for wedges open over 225° to 315° (within
  0.25 dB); a thin screen against Maekawa's chart (within 1.5 dB), the same from either
  side; and the early level across the shadow boundary of an L-shaped room, in
  millimetre steps, continuous within 0.25 dB with first-order reflections where
  specular paths alone drop by more than 5 dB, and across the boundary of a reflection
  off one of the corner's walls within 0.05 dB; over a 1 m thick barrier, the path
  over both corners against the exact BTM solution (within 1 dB, and 0.2 dB from
  500 Hz up, the same both ways round), and the level as the source and the listener
  rise through the plane of its top, continuous within 1 dB (0.1 dB from 1 kHz up); a
  thin screen's diffracted wave in its lit zone, all four terms of the UTD within 0.2 %;
- early reflections in pressure at low frequencies: an impulse response and the
  echogram it came from in a 7.3 x 5.1 x 3 m box, early part alone, G within 0.02 dB,
  C50 within 0.05 dB and Ts within 0.3 ms at 63 and 125 Hz (G 2.1 dB and C50 1.6 dB
  apart when the paths add in energy); a source and a listener 0.3 m above a floor 10 m apart, the direct
  sound and the floor reflection 5.5 dB above the free-field level over a hard floor
  and 4 dB below it over an absorbing one at 63 and 125 Hz, the echogram within 0.3 dB
  of the paths' pressures added; the phase of the reflection factor, none off a hard
  surface and half a turn off an absorbing one at grazing incidence; and a thin
  screen's diffracted wave, with all four terms of the UTD added in pressure, to
  1e-9;
- the wave solver, with both stencils where both apply: the modes of a rigid box
  (within 0.6 Hz), both stencils' modes along a tube at 12, 8 and 5 cells per
  wavelength as slow as their dispersion relation along an axis says (within 2e-5,
  the 19-point stencil's error less than half the 7-point one's), the 19-point
  stencil stable for 30,000 steps in a rotated room with a cube and a screen in it,
  both stencils' highest modes 2⁻¹⁶ inside the stability limit (kicked, within
  1.1 × 256 times the kick over 20,000 steps, rounded as the update rounds),
  the reflection off the end
  of a duct for four admittances (within 0.005, and below 0.05 for a matched end, and
  off the inner face of a wall thinner than a cell), and off an end whose admittance
  resonates (within 0.012 of the grid's theory from 30 to 250 Hz); admittances fitted to
  four kinds of material (within 0.01 of their absorption in every band, passive); the
  free-field level of the 63 Hz band (within 1 dB of the rays), the arrival time and
  direction of the direct sound, open models and thin screens (stable, and the sound
  leaves), the matched layer round an open model (what comes back 40 dB down or more,
  against a grid too large for anything to come back; stable for 9,500 steps), one run
  for bands with their own absorption, and the level of the late part across the
  crossover to the rays (flat within 1 dB in sixth octaves);
- the ray tracer against runs stored before its tree was rebuilt
  (`tests/reference/tracer.npz`): the faces and distances the tree finds, the specular
  paths, and every kind of late run on the CPU to single precision, so a change that
  moves a result is seen; the same runs traced on a GPU (in CI, Mesa's software Vulkan
  driver) land within the spread of the CPU's own results over seeds, and the GPU's ray
  searches find the CPU's faces; outdoors the GPU's chords through the atmosphere are
  the CPU's ray for ray, and what its rays bent by the weather or scattered by plants
  bring to each listener agrees within the noise (see
  [Outdoor scenes on the GPU](#outdoor-scenes-on-the-gpu));
- the leak finder: a closed model keeps every ray; a box without its ceiling reports
  one gap of 30 m² (6 x 5 m) with every ray through it, at the ceiling's height; the
  example classroom's two holes, the missing window pane in the east wall (3.6 m², 3.0 x
  1.2 m) and the 3 cm slit along the ceiling's edge (0.27 m²), each with the rays that
  crossed it inside it; a ceiling that lets sound through counted as an opening, not a
  leak; a source outside a closed model named; and no leak in the other examples;
- walkthroughs: the places along a walk every 25 cm, facing forward and turning the
  shorter way round at the set pace; crossfade weights that add up to one and are
  continuous; the late parts of the places between those traced mixed in proportion to
  the distance; one response everywhere equal to the plain convolution (to 1e-12); a
  walking listener, and a walking source heard at a listener; and the gallery walk
  rendered through its places;
- models read from files: a room whose walls' edges run past the floor's corners
  (T-junctions, with or without the zero-area slivers SketchUp exports) cleaned into a
  closed room of the right volume and area; two rooms touching along an edge closed; a
  face given twice leaving the model open, with its odd edges counted; points inside a
  model whichever way its faces point; an IFC office read as one room from its space
  boundaries (closed, 75.6 m³) or as every element, in the file's units; a Collada room
  in inches with its up axis as written, and instanced components placed by their
  nodes; and materials from CATT-Acoustic files and ODEON libraries, the latter as
  ODEON writes it (Windows encodings and six-band tables too);
- outdoor scenes: over a flat ground under the open sky, exactly two paths, the direct
  sound and the ground's reflection, at their distances and with the reflection's
  energy (absorption, spreading and air) to rounding, from both early methods; a
  barrier's top and ends as its only diffracting edges, with the shortest path over its
  top; no edge at the inside corner of an L-shaped building, whichever way its
  footprint runs; points inside buildings refused and left out of coverage maps; and
  responses that run `ir_length_s` past the latest direct sound;
- grounds with an impedance: Miki's and Delany and Bazley's models at their constants,
  the two-parameter model's closed form and a layer's impedance; the spherical wave
  reflected whole off a hard ground, as the plane wave far from grazing and far away,
  and within 0.4 dB of the exact solution over grass, a thin layer of snow and gravel;
  the early part over grass within 0.05 dB of the spherical wave in every octave band,
  however large the ground; the points of a line source added in energy; a ground zone
  far past a reflection's Fresnel zone changing nothing (to 1e-9), one under all of it
  the same as a ground of its own, and the change in between smooth; and the wave
  solver's fit to an impedance within 0.05 of its reflection factor, from normal to
  near grazing incidence, and within 0.01 of its absorption;
- vegetation: ISO 9613-2's dense foliage to its table, a rigid cylinder's scattering
  against the exact series' small and large limits, and the depth through belts, hedges
  mitred at their bends and crowns; the early paths losing exp(-Σ μ l) along their legs
  to 1e-9; the late rays losing what a slab absorbs as Beer-Lambert has it (within
  0.3 %), and upright stems meeting sound by its horizontal share (sound going up
  through a layer of them barely touches them); two bands traced with one set of rays
  as each traced alone
  (within 3 %); the rain of foliage and of stems against the detector spheres; and
  scenes without vegetation unchanged; leaves given by their area and width against a
  thin disc's closed forms (the dipole of a small one to 0.2 %, twice the mean
  projected area of a large one, its mean cosine towards 1/2) and a rigid wall's
  boundary-layer absorption (to 1e-12), per band as the mean over its frequencies,
  wider leaves of the same area taking more where their size nears the wavelength, and
  the early paths through a belt of them losing what the medium says (within 5 %);
- the wave solver in wind and temperature: the grid's own speed of sound in every cell
  giving the run without a speed per cell, to the bit (both stencils, with walls of
  frequency-dependent admittance); 0.8 times it giving sound that comes 1.25 times as
  late and as loud, its velocity as large next to its pressure; sound bent by a speed of
  sound that grows or falls by 65 m/s over 60 m arriving from 5 degrees above or below,
  within 12 % of the eigenrays' angle; and the matched layer stable where the speed of
  sound halves with height and jumps across the grid;
- sweep measurements of synthetic rooms through a distorting loudspeaker, with
  noise and interface latency: from 250 Hz up, the room's response comes back with
  errors 30 dB below it and T30 within 3 %; with a noise floor 55 dB down, T30 stays
  within 5 % of the noise-free value.
- outdoor results: the level relative to free field over grass against the exact
  spherical-wave reflection (within 0.05 dB from 1 kHz up, 0.75 dB in the
  band-filtered dip at 500 Hz), and over an absorbing ground that of free field; a
  barrier's insertion loss the level relative to free field with its sign turned,
  with a reflection back on the source's side that of its image source; the two runs
  sharing their seed, and maps giving the same at their points;
- a terrain: heights read between the grid's nodes bilinearly and past its edge as the
  edge's, from ESRI ASCII grids (north row first) and XYZ points in any order; a flat
  terrain giving the flat ground's early paths to the bit and a plane slope the
  reflection off the image source in it; the sound over a ridge the exact wedge's (UTD
  against the Biot-Tolstoy-Medwin solution, within 0.1 dB over the source's and the
  listener's slopes); a smooth hill's crest bending the sound, the level continuous
  as a listener climbs out of its shadow, over two hills each crest bending it in turn,
  a plane slope screening nothing and a building on the string stopping the path;
  buildings, barriers and vegetation standing on the terrain, zones taking its
  triangles, the extent keeping to it, coverage maps and insertion losses following
  it, footprints read from GeoJSON polygons and multipolygons, their holes built over
  and said so, and `inspect` describing the terrain and the heights over the ground.

## Cross-check against pyroomacoustics

[pyroomacoustics](https://github.com/LCAV/pyroomacoustics) 0.10.1 also simulates rooms
with image sources for the early part and stochastic ray tracing for the rest, which
makes it the open-source program closest to this one. Both simulated the same four
rooms:

- **specular**: an 8 x 6 x 3 m box with painted concrete walls, a carpeted floor and an
  absorbing tile ceiling, every surface reflecting specularly (no scattering);
- **diffuse**: the same box with every surface fully diffusing (scattering 1);
- **example**: the same box with the example scene's scattering: walls 0.05, floor 0.1,
  ceiling 0.2;
- **hall**: the fan-shaped lecture hall of `examples/lecture_hall` (1008 m³) with an
  audience floor (scattering 0.7), plasterboard ceiling, wooden stage wall and concrete
  side and back walls (scattering 0.1 to 0.15).

Both programs got the same absorption and scattering per octave band, the same air
absorption (ISO 9613-1 at 20 °C and 50 %, from this package), image sources up to the
3rd order and 100,000 rays, with one omnidirectional source and four listeners per room.
pyroomacoustics draws its rays afresh on every run, so each room ran five times in it:
the tables give the mean of the five and the spread between the highest and the lowest.
Both scatter vector-based. pyroomacoustics has no scattering from the size of a surface,
so this package left it out for the comparison (`from_size: false`) and ran each room
once more with its default, which adds it. Both sets of impulse responses were analysed
by the same code (`acoustic_spaces.parameters.from_impulse_response`), and EDT, T30 and
C80 were averaged over the four listeners, in the octave bands from 125 Hz to 4 kHz.

### How the two programs differ

Reading the ray tracer of pyroomacoustics 0.10.1 (`libroom_src/room.cpp`) shows two
differences besides the scattering of a surface's size:

- **How the listener receives scattered energy.** Both programs send the scattered share
  of every reflection straight to each listener that can see the reflection point
  ("diffuse rain"). Here the detector sphere then counts only the rest of a ray's
  energy. pyroomacoustics counts all of it at the sphere, so the scattered share reaches
  the listener twice, which raises the reverberant level, lengthens EDT and lowers C80.
- **One direction for all bands.** pyroomacoustics turns a ray by the mean of the bands'
  scattering coefficients; this package traces each band with rays of its own where the
  coefficients differ. In these four rooms they are the same in every band, so this
  makes no difference here.

The cross-check script also traces the total sound energy in each room without any
listener, with either model, and fits T30 to the same stretch of its decay as the
listeners' T30 ("room energy" rows below).

### Results

The figures show EDT, T30 and C80 per band for both programs, the spread of
pyroomacoustics' runs shaded, this package's default with
the scattering of the surfaces' size, the decay of the room's total energy under each
model and, where a diffuse field makes them apply, Eyring's reverberation time and
Barron's revised theory.

**Specular box.** The programs agree: T30 within 3 %, EDT within 7 % and C80 within
1 dB. Rays that travel parallel to the floor never meet the absorbing
ceiling, so the decay is many times longer than Eyring's estimate in both. Even a
smooth wall is only 3 m high, though, and scatters what reaches it from far away or at a
grazing angle: with that, T30 comes out 2.7 to 4.7 s instead of 3 to 15 s.

| | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---:|---:|---:|---:|---:|---:|
| EDT (s), acoustic_spaces | 7.31 | 6.37 | 3.54 | 3.00 | 2.33 | 1.33 |
| EDT (s), pyroomacoustics | 7.27 | 5.95 | 3.56 | 2.94 | 2.21 | 1.28 |
| EDT (s), pyroomacoustics' spread over 5 runs | 1.31 | 0.49 | 0.36 | 0.14 | 0.13 | 0.03 |
| EDT difference | -1 % | -7 % | +1 % | -2 % | -5 % | -4 % |
| T30 (s), acoustic_spaces | 15.10 | 14.03 | 8.31 | 6.31 | 4.98 | 3.05 |
| T30 (s), pyroomacoustics | 15.32 | 13.75 | 8.23 | 6.27 | 5.00 | 3.15 |
| T30 (s), pyroomacoustics' spread over 5 runs | 0.51 | 0.21 | 0.11 | 0.07 | 0.04 | 0.04 |
| T30 difference | +1 % | -2 % | -1 % | -1 % | +1 % | +3 % |
| C80 (dB), acoustic_spaces | -2.9 | -1.7 | 0.1 | 1.8 | 3.1 | 5.0 |
| C80 (dB), pyroomacoustics | -2.3 | -1.0 | 0.9 | 2.0 | 3.2 | 5.2 |
| C80 (dB), pyroomacoustics' spread over 5 runs | 2.6 | 1.5 | 0.9 | 0.7 | 0.4 | 0.2 |
| C80 difference | +0.6 dB | +0.7 dB | +0.7 dB | +0.1 dB | +0.1 dB | +0.2 dB |
| T30 (s), acoustic_spaces with surface size (default) | 3.91 | 4.43 | 4.58 | 4.66 | 4.10 | 2.69 |
| T30 (s) of the room's energy, vector-based scattering | 15.64 | 13.95 | 7.29 | 6.29 | 4.96 | 2.87 |
| T30 (s) of the room's energy, with surface size | 4.23 | 5.36 | 4.76 | 5.14 | 4.29 | 2.64 |
| Eyring (s) | 0.83 | 0.69 | 0.52 | 0.37 | 0.30 | 0.25 |

![Specular box](validation/crosscheck_specular.png)

**Diffuse box.** T30 agrees within 5 %; this package stays within 8 % of the room's
energy decay and within 10 % of Eyring's estimate, pyroomacoustics within 11 and 15 %.
From 1 kHz up, pyroomacoustics gives 13 to 21 % longer EDT and 1.7 to 2.2 dB lower C80
than this package, as its
double-counted scattered energy predicts. Barron's revised theory sides with this
package: computed from the simulated energies, as `simulate` prints them, C80 is 0.6 to
1.0 dB below the theory in every band, where pyroomacoustics' (from its impulse
responses) is 1.1 to 3.2 dB below it. From the impulse responses this package's is about
0.5 and 1.2 dB lower again at 250 and 500 Hz (2.0 dB below the theory at 500 Hz), because
the direct sound and the first floor and ceiling reflections interfere within those
octave bands. Surfaces that diffuse fully
scatter no more for their size, so the default gives the same.

| | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---:|---:|---:|---:|---:|---:|
| EDT (s), acoustic_spaces | 0.91 | 0.70 | 0.61 | 0.38 | 0.31 | 0.26 |
| EDT (s), pyroomacoustics | 0.83 | 0.69 | 0.57 | 0.43 | 0.37 | 0.32 |
| EDT (s), pyroomacoustics' spread over 5 runs | 0.13 | 0.09 | 0.04 | 0.02 | 0.03 | 0.02 |
| EDT difference | -8 % | -1 % | -6 % | +13 % | +18 % | +21 % |
| T30 (s), acoustic_spaces | 0.75 | 0.67 | 0.53 | 0.40 | 0.30 | 0.26 |
| T30 (s), pyroomacoustics | 0.71 | 0.65 | 0.54 | 0.41 | 0.31 | 0.26 |
| T30 (s), pyroomacoustics' spread over 5 runs | 0.04 | 0.04 | 0.03 | 0.02 | 0.01 | 0.01 |
| T30 difference | -5 % | -4 % | +2 % | +4 % | +4 % | +2 % |
| C80 (dB), acoustic_spaces | 4.9 | 5.6 | 7.1 | 12.8 | 15.6 | 19.0 |
| C80 (dB), pyroomacoustics | 4.2 | 5.4 | 7.4 | 11.0 | 13.9 | 16.8 |
| C80 (dB), pyroomacoustics' spread over 5 runs | 3.1 | 1.9 | 1.2 | 0.7 | 1.0 | 0.3 |
| C80 difference | -0.7 dB | -0.2 dB | +0.3 dB | -1.8 dB | -1.7 dB | -2.2 dB |
| C80 (dB), Barron's revised theory | 5.3 | 6.8 | 9.1 | 13.6 | 17.1 | 19.8 |
| T30 (s), acoustic_spaces with surface size (default) | 0.75 | 0.67 | 0.53 | 0.40 | 0.30 | 0.26 |
| T30 (s) of the room's energy, vector-based scattering | 0.78 | 0.66 | 0.52 | 0.37 | 0.30 | 0.26 |
| T30 (s) of the room's energy, with surface size | 0.78 | 0.66 | 0.52 | 0.37 | 0.30 | 0.26 |
| Eyring (s) | 0.83 | 0.69 | 0.52 | 0.37 | 0.30 | 0.25 |

![Diffuse box](validation/crosscheck_diffuse.png)

**Example box.** With smooth walls (scattering 0.05), T30 agrees within 4 %, EDT within
7 % and C80 within 1.3 dB. Up to version 2 this package scattered a ray with the
coefficient as probability, and pyroomacoustics' T30 came out 19 to 42 % shorter. The
walls' size halves T30 at 125 and 250 Hz (0.86 s against 1.82 s) and shortens it by
8 to 22 % from 500 Hz up; each run matches the room-energy decay of its model within 11 %.

| | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---:|---:|---:|---:|---:|---:|
| EDT (s), acoustic_spaces | 1.27 | 1.29 | 1.19 | 1.09 | 1.08 | 0.92 |
| EDT (s), pyroomacoustics | 1.30 | 1.20 | 1.12 | 1.14 | 1.08 | 0.86 |
| EDT (s), pyroomacoustics' spread over 5 runs | 0.20 | 0.05 | 0.23 | 0.13 | 0.13 | 0.12 |
| EDT difference | +2 % | -7 % | -6 % | +4 % | +0 % | -6 % |
| T30 (s), acoustic_spaces | 1.82 | 1.76 | 1.57 | 1.52 | 1.44 | 1.23 |
| T30 (s), pyroomacoustics | 1.81 | 1.81 | 1.64 | 1.56 | 1.45 | 1.24 |
| T30 (s), pyroomacoustics' spread over 5 runs | 0.27 | 0.12 | 0.09 | 0.14 | 0.15 | 0.07 |
| T30 difference | +0 % | +3 % | +4 % | +3 % | +1 % | +1 % |
| C80 (dB), acoustic_spaces | 2.7 | 2.5 | 3.3 | 5.1 | 5.1 | 6.3 |
| C80 (dB), pyroomacoustics | 2.9 | 3.4 | 4.6 | 4.8 | 5.2 | 6.5 |
| C80 (dB), pyroomacoustics' spread over 5 runs | 1.2 | 0.7 | 0.6 | 0.7 | 0.7 | 0.7 |
| C80 difference | +0.2 dB | +0.8 dB | +1.3 dB | -0.3 dB | +0.1 dB | +0.2 dB |
| T30 (s), acoustic_spaces with surface size (default) | 0.86 | 0.97 | 1.22 | 1.27 | 1.25 | 1.13 |
| T30 (s) of the room's energy, vector-based scattering | 1.65 | 1.64 | 1.53 | 1.49 | 1.42 | 1.22 |
| T30 (s) of the room's energy, with surface size | 0.90 | 0.93 | 1.29 | 1.28 | 1.36 | 1.18 |
| Eyring (s) | 0.83 | 0.69 | 0.52 | 0.37 | 0.30 | 0.25 |

![Example box](validation/crosscheck_example.png)

**Lecture hall.** T30 agrees within 3 % and EDT within 5 %; pyroomacoustics' C80 is up to
1.2 dB lower (at 125 Hz), the double-counted scattered energy again. The surfaces' size
shortens T30 by 4 to 16 %, in line with the room-energy decay of that model within 6 %.

| | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---:|---:|---:|---:|---:|---:|
| EDT (s), acoustic_spaces | 1.95 | 1.73 | 1.49 | 1.41 | 1.46 | 1.21 |
| EDT (s), pyroomacoustics | 1.90 | 1.80 | 1.43 | 1.47 | 1.42 | 1.15 |
| EDT (s), pyroomacoustics' spread over 5 runs | 0.30 | 0.23 | 0.12 | 0.05 | 0.10 | 0.07 |
| EDT difference | -2 % | +4 % | -4 % | +4 % | -3 % | -5 % |
| T30 (s), acoustic_spaces | 1.97 | 1.93 | 1.89 | 1.90 | 1.85 | 1.59 |
| T30 (s), pyroomacoustics | 1.96 | 1.95 | 1.90 | 1.96 | 1.87 | 1.59 |
| T30 (s), pyroomacoustics' spread over 5 runs | 0.12 | 0.14 | 0.10 | 0.03 | 0.06 | 0.06 |
| T30 difference | -1 % | +1 % | +0 % | +3 % | +1 % | +0 % |
| C80 (dB), acoustic_spaces | 0.2 | -0.1 | 1.6 | 3.0 | 3.0 | 3.9 |
| C80 (dB), pyroomacoustics | -1.0 | -0.4 | 1.7 | 2.5 | 2.8 | 3.7 |
| C80 (dB), pyroomacoustics' spread over 5 runs | 2.2 | 0.9 | 1.3 | 0.4 | 0.4 | 0.2 |
| C80 difference | -1.2 dB | -0.3 dB | +0.1 dB | -0.6 dB | -0.2 dB | -0.1 dB |
| T30 (s), acoustic_spaces with surface size (default) | 1.88 | 1.74 | 1.59 | 1.70 | 1.74 | 1.53 |
| T30 (s) of the room's energy, vector-based scattering | 1.97 | 1.99 | 1.92 | 1.96 | 1.90 | 1.58 |
| T30 (s) of the room's energy, with surface size | 1.84 | 1.84 | 1.55 | 1.69 | 1.77 | 1.54 |
| Eyring (s) | 1.99 | 1.89 | 1.13 | 0.99 | 0.88 | 0.86 |

![Lecture hall](validation/crosscheck_hall.png)

### Conclusions

- With the same scattering model, the programs agree within 5 % on T30 in all four
  rooms, from purely specular to fully diffusing surfaces, and within 8 % on EDT where
  pyroomacoustics' double counting does not lengthen it. That is close to the 5 % and
  1 dB the plan aimed for. With 100,000 rays, pyroomacoustics' own runs differ by up to
  15 % in T30, 21 % in EDT and 3.1 dB in C80, most often at 125 Hz, which is why the tables
  average five of them; this package's numbers move by a few percent with the seed.
- Where the theory is clear (the diffuse box), this package agrees with Eyring (T30
  within 10 %) and with Barron's revised theory (C80 from the energies within 1.0 dB);
  pyroomacoustics counts the scattered energy twice.
- The scattering of a surface's size, which pyroomacoustics lacks, shortens the decay of
  rooms with smooth walls a lot at low frequencies and less at high ones. Measurements,
  not another program, decide whether it helps: in the two measured rooms below it does.

## Measured rooms (BRAS)

The *Benchmark for Room Acoustical Simulation* (BRAS; Aspöck, Brinkmann, Ackermann,
Weinzierl and Vorländer, 2020,
[DOI 10.14279/depositonce-6726.3](https://doi.org/10.14279/depositonce-6726.3), CC BY-SA
4.0) measured real rooms and published their models, the absorption and scattering of
their surfaces, and the impulse responses. Two of its rooms are in `validation/bras` as
scenes (its README says how they were made):

- **CR2**, a seminar room of 146 m³ with five surface materials;
- **CR3**, a chamber music hall of 3331 m³ with upholstered seating, stage panels and
  structured plaster, seven materials.

In each, a dodecahedron loudspeaker stood at two positions and an omnidirectional
microphone at five, giving ten measured responses. Each scene was simulated as it stands,
with this package's defaults: ray tracing up to the 3rd order, first-order edge
diffraction, 50,000 late rays per band and vector-based scattering with the scattering
of each surface's size, in the air measured on the day. Simulated and measured
parameters were compared pair by pair in just noticeable differences
(`acoustic_spaces.compare`, the same JNDs as `acoustic-spaces compare`) and averaged over
the ten pairs. The measured responses were analysed by this package too. Its octave
values, averaged over the ten pairs, are within 3 % of the mean of the three third-octave
averages BRAS publishes for T20, within 5 % for EDT and within 0.7 dB for C80, from
500 Hz to 4 kHz (the largest differences are in the hall at 500 Hz: EDT 4.3 % lower,
C80 0.6 dB higher; elsewhere within 3.2 % and 0.5 dB). Below that the octave holds three
third-octave bands whose values differ a lot, so averaging them is not the same analysis.
G is left out because the responses have no free-field reference.

The tables and plots in this section were made with version 4.0.1 on 27 and 28 October
2026, on 4 cores of a cloud machine without a GPU, where the run times below were
measured. 4.0.0 could not run either room at its defaults (its early stage asked for
16.4 GiB at once in CR3 and 19.6 GiB in CR2); 4.0.1 runs the seminar room in 1.1 GiB
(1.2 GiB in third-octave bands) and the hall in 10.0 GiB (10.4 GiB
calibrated, 11.7 GiB in third-octave bands), most of it the edge fade's at one listener (see
[Memory](simulation.md#memory)).

BRAS gives two sets of absorption coefficients. The **fitted** set was adjusted until
Eyring's formula gave the measured reverberation time; the scenes use it. The
**initial** set is BRAS's estimate before that, from tables and measurements of the
materials, which is the blind case (`--materials initial`).

Mean absolute difference from 125 Hz to 4 kHz in JNDs, and how many of the 60 pair and
band values are within one JND:

| | CR2 fitted | CR3 fitted | CR2 initial | CR3 initial |
|---|---:|---:|---:|---:|
| EDT | 1.0 (44) | 2.1 (20) | 5.9 (6) | 4.6 (9) |
| T20 | 1.0 (41) | 2.0 (10) | 6.5 (1) | 5.7 (12) |
| T30 | 1.0 (43) | 1.8 (9) | 6.5 (0) | 5.7 (11) |
| C50 | 1.0 (41) | 1.2 (33) | 1.6 (26) | 1.6 (21) |
| C80 | 0.7 (51) | 1.0 (40) | 1.6 (28) | 1.3 (27) |
| D50 | 1.0 (40) | 1.4 (31) | 1.6 (24) | 1.7 (20) |
| Ts | 0.8 (42) | 0.9 (33) | 3.2 (13) | 2.1 (24) |

![CR2, fitted absorption](validation/bras_CR2_fitted.png)

**Seminar room, fitted absorption.** From 500 Hz to 4 kHz the reverberation time is
within 2 % of the measured one (T20 1.92 s against 1.94 s at 1 kHz), well within one JND,
and C50, C80 and D50 are within one JND from 500 Hz up. At 250 Hz and 8 kHz the simulated
decay is 13 and 18 % short, and so is Eyring's formula with the same absorption and air (1.48 s
against 1.72 s measured at 250 Hz, 0.93 s against 1.11 s at 8 kHz), so there the fitted
coefficients, averaged onto octaves, do not give the measured decay even in a diffuse
field. At 8 kHz they assume less air absorption than ISO 9613-1 gives at the octave's
centre frequency; a measured decay follows the octave's slowest part, its lowest third
(6.3 kHz). Simulated in third-octave bands, which keep both apart, the two bands come
within 4.1 % of the measurement (see the third-octave results below).

![CR3, fitted absorption](validation/bras_CR3_fitted.png)

**Chamber music hall, fitted absorption.** The reverberation time comes out 7 to 10 % long
from 250 Hz to 4 kHz (T20 1.47 s against 1.34 s at 1 kHz), 1.3 to 1.9 JNDs, and 11 and
17 % long at 63 and 125 Hz. EDT is within 3 % at 250 and 500 Hz and 6 to 14 % long from
1 to 4 kHz, but 19 and 13 % short at 63 and 125 Hz, where the early part is stronger
than measured (see the next paragraph). From 125 Hz up, C50,
C80 and D50 are within about one JND, mostly slightly low, as a slightly long decay
predicts. Eyring's formula with the same absorption and air is within 5.1 % of the
measurement from 63 Hz to 4 kHz, as the fitting intends, so in the
model sound still reaches the absorbing seating a little less often than Eyring's
diffuse field assumes, and than in the real hall, whose details the simplified BRAS
model smooths away.

**The hall's early part at low frequencies.** The plots of 15 October 2026 had the hall's
EDT about 23 and 19 % long at 63 and 125 Hz and C80 about 1.5 and 1.3 dB low: their early
part was weaker. Traced again at LS1's five positions without edge diffraction, that
day's code and this version give the same early energy, bit for bit, with every band in
energy as that day's code had them (with 63 and 125 Hz in pressure, this version's
default, 0.5 and 0.2 dB more). With it (`first_order`) this version's early energy is
6.6 and 5.8 dB higher at 63 and 125 Hz, 2.3 dB at 250 Hz and 1.0 dB at 500 Hz (means over
the five positions; 4.7 to 9.8 dB at 63 Hz), and as high or higher with every band in
energy (`coherent_max_band_hz: null`), so the reflections' phases are not the cause. The
late part is the same. Without the
[fade at edges](#reflections-that-fade-at-edges-against-the-exact-solution), which
4.0.0 added, it is 3.5 and 1.4 dB higher (1.0 to 6.4 dB at 63 Hz): the fade adds 3.1 and
4.4 dB of the change. At 125 Hz C80 now comes within 0.3 dB of the measurement;
at 63 Hz the early part is too strong (C80 1.0 dB high, EDT 3.5 JNDs short).

**What scattering does.** Up to version 2 this package scattered a ray with the
coefficient as probability, and from the surface's own coefficient alone. The hall then
came out 20 to 32 % long, EDT 11 to 24 %, and the seminar room 0 to 10 % long from 250 Hz
to 4 kHz. Tracing the total sound energy in each room without listeners (as in the
cross-check above, each band with rays of its own) and fitting T20 to its decay shows
what each part of the new model does:

| T20 (s) | 63 Hz | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz | 8 kHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CR2 measured | 1.56 | 1.34 | 1.72 | 2.00 | 1.94 | 1.75 | 1.54 | 1.11 |
| CR2 room energy, random scattering (version 2) | 1.51 | 1.49 | 1.70 | 2.17 | 2.06 | 1.83 | 1.60 | 0.97 |
| CR2 room energy, vector-based scattering | 1.57 | 1.41 | 1.51 | 1.82 | 1.79 | 1.66 | 1.44 | 0.90 |
| CR2 room energy, vector-based with surface size (default) | 1.69 | 1.49 | 1.50 | 1.97 | 1.92 | 1.76 | 1.53 | 0.91 |
| CR2 Eyring | 1.45 | 1.34 | 1.48 | 2.00 | 1.95 | 1.76 | 1.59 | 0.93 |
| CR3 measured | 1.77 | 1.57 | 1.46 | 1.28 | 1.34 | 1.30 | 1.03 | 0.78 |
| CR3 room energy, random scattering (version 2) | 2.11 | 2.01 | 1.77 | 1.60 | 1.63 | 1.56 | 1.18 | 0.74 |
| CR3 room energy, vector-based scattering | 2.11 | 1.95 | 1.69 | 1.51 | 1.52 | 1.43 | 1.11 | 0.74 |
| CR3 room energy, vector-based with surface size (default) | 2.03 | 1.89 | 1.63 | 1.40 | 1.47 | 1.40 | 1.12 | 0.74 |
| CR3 Eyring | 1.79 | 1.65 | 1.49 | 1.28 | 1.33 | 1.33 | 1.06 | 0.71 |

From 500 Hz to 4 kHz, vector-based scattering alone takes the hall from 15 to 25 % long
down to 8 to 18 %, but makes the seminar room 5 to 9 % short. Adding the scattering of
each surface's size brings the seminar room within 2 % of the measurement (and within
4 % of Eyring's diffuse field), and the hall to 8 to 10 % long: one model now fits both rooms better than
either did. The simulated responses decay like these room-energy curves, within 4.3 %
in every band. The rest of the hall's gap is the model hall itself, as the paragraph above
says. Fitting the absorption with this tracer instead of Eyring's formula takes up that
gap (see "Calibrated absorption" below).

**Third-octave bands.** BRAS gives its coefficients in third octaves. Averaged onto
octaves, an octave's three thirds decay as one, at their mean absorption, while three
decays at different rates decay together more slowly than that, as the measured octave
does. Simulated with `simulation.bands: third_octave`, which keeps the thirds apart
(`bras_validation.py --bands third_octave`), T20 in each octave band:

| T20 (s) | 63 Hz | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz | 8 kHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CR2 measured | 1.56 | 1.34 | 1.72 | 2.00 | 1.94 | 1.75 | 1.54 | 1.11 |
| CR2 octave bands (default) | 1.69 | 1.49 | 1.50 | 1.97 | 1.92 | 1.76 | 1.53 | 0.91 |
| CR2 third-octave bands | 1.77 | 1.51 | 1.65 | 1.98 | 1.92 | 1.78 | 1.51 | 1.07 |
| CR3 measured | 1.77 | 1.57 | 1.46 | 1.28 | 1.34 | 1.30 | 1.03 | 0.78 |
| CR3 octave bands (default) | 1.96 | 1.83 | 1.56 | 1.40 | 1.47 | 1.40 | 1.10 | 0.72 |
| CR3 third-octave bands | 2.00 | 1.88 | 1.57 | 1.40 | 1.46 | 1.40 | 1.13 | 0.80 |

The seminar room's decay at 250 Hz and 8 kHz, 13 and 18 % short in octave bands, comes
within 4.1 % of the measurement: there its thirds' absorption, and at 8 kHz the air's,
differ most. The hall's 8 kHz goes from 8 % short to 3 % long. From 500 Hz to 2 kHz T20
changes by 0.02 s at most. At 63 and 125 Hz both rooms come out 1 to 5 % longer than in
octave bands, and the hall at 4 kHz 10 % long instead of 7 %. Over 125 Hz to 4 kHz the seminar
room's mean T30 difference goes from 1.0 to 0.9 JNDs; the hall's goes from 1.8 to 2.1,
since its decay was long already and third octaves lengthen it. A third-octave run takes
9 minutes for the seminar room instead of 4, and 3.6 hours for the hall instead of 1.5.

![CR2, initial absorption](validation/bras_CR2_initial.png)

![CR3, initial absorption](validation/bras_CR3_initial.png)

**Initial absorption.** With the absorption estimated before fitting, the decay at 63
and 125 Hz comes out 1.5 to 2.2 times as long as measured in both rooms, and at 250 Hz in
the hall too (T20 2.71 s against 1.34 s in CR2 at 125 Hz). The initial coefficients have
far less absorption there than the fitted ones. In real rooms much of it comes from
structures that flex, such as panels, windows and doors, which tables of materials
rarely capture. From 500 Hz to 4 kHz the seminar room's decay is 13 to 29 % short, and at
2 and 4 kHz the hall's is 7 to 18 % short. The simulation follows Eyring's formula with
the same coefficients within 15 % in every band of both rooms (within 3 % from 250 Hz up
in the seminar room), so the error is in the coefficients. This is the error to expect from tabulated coefficients alone, before a
room has been measured.

**Calibrated absorption.** `--calibrate LS1` first fits the absorption of every surface
(`acoustic-spaces calibrate`, see [Measurements](measurements.md#fitting-materials-to-measurements)). It fits to the T30, EDT and C80 of the
five responses of the first loudspeaker position, starting from the initial set, with
10,000 rays per band while fitting. Then it simulates the room with what the fit found,
as above, and compares the five responses of the second position, which the fit did
not see. The table gives the mean absolute difference from 125 Hz to 4 kHz in JNDs over
those five responses, with how many of their 30 pair and band values are within one
JND. The fitted and initial columns repeat the table above, which averages all ten
responses:

| | CR2 calibrated | CR2 fitted | CR2 initial | CR3 calibrated | CR3 fitted | CR3 initial |
|---|---:|---:|---:|---:|---:|---:|
| EDT | 1.1 (20) | 1.0 | 5.9 | 1.5 (15) | 2.1 | 4.6 |
| T20 | 0.9 (23) | 1.0 | 6.5 | 1.2 (18) | 2.0 | 5.7 |
| T30 | 1.0 (22) | 1.0 | 6.5 | 0.7 (25) | 1.8 | 5.7 |
| C50 | 1.1 (20) | 1.0 | 1.6 | 1.2 (15) | 1.2 | 1.6 |
| C80 | 0.7 (25) | 0.7 | 1.6 | 0.8 (20) | 1.0 | 1.3 |
| D50 | 1.1 (20) | 1.0 | 1.6 | 1.3 (13) | 1.4 | 1.7 |
| Ts | 0.7 (26) | 0.8 | 3.2 | 0.8 (20) | 0.9 | 2.1 |

![CR2, calibrated from the initial absorption](validation/bras_CR2_initial_calibrated.png)

![CR3, calibrated from the initial absorption](validation/bras_CR3_initial_calibrated.png)

The initial estimates made the decay up to twice as long. From them, calibration brings
the seminar room to about one JND on responses it did not fit, as BRAS's fitted set
does. In the hall it does better than the fitted set: T30 is within 0.7 JNDs on
average, with 25 of the 30 values within one JND, against 1.8 with the fitted set, T20
within 1.2 against 2.0 and EDT 1.5 against 2.1. The fit uses the model's own rays
rather than Eyring's formula, so it takes up what the model hall's sound misses of the
seating.

In the seminar room the fit found nearly BRAS's fitted set, though it started from the
initial one. The windows absorb 0.28 and 0.22 at 63 and 125 Hz (BRAS fitted 0.25 and
0.19, initial 0.13 and 0.10), and every other value is within 0.025 of BRAS's. In
the hall it found more absorption than BRAS in the plaster at 125 and 250 Hz (0.20 and
0.20 against 0.11 and 0.15) and in the seating at 2 kHz (0.63 against 0.41), and less
in the seating at 125 and 250 Hz (0.11 and 0.22 against 0.21 and 0.37) and in the
structured plaster at 2 kHz (0.09 against 0.14).

These are what a model needs to match the room, not what each surface absorbs. With one
loudspeaker position and five microphones, the uncertainty the fit reports is above 0.2
for most materials and bands in the hall, and 0.38 or more for all of them in the seminar
room. The
measurements fix each band's total absorption, and the proportions between materials
come from the initial estimates. Telling surfaces apart takes responses that hear them
differently, such as sources and microphones near different surfaces.

Some differences remain:

- **250 Hz in the seminar room.** T30 is 3.1 JNDs short on the second loudspeaker's
  responses, though 1.2 off on average on the first's. The two positions measured 1.70 and
  1.90 s on average. This is near the room's Schroeder frequency, about 230 Hz, where a
  decay depends on where it is measured.
- **8 kHz in the seminar room.** The decay stays 2 JNDs short even with surfaces that
  absorb next to nothing, because in octave bands the air absorbs at the octave's centre
  frequency (see "Third-octave bands" above).
- **63 Hz in the seminar room.** EDT stays 5.8 JNDs long. Below the Schroeder frequency
  a geometric model gives an average, not the room's modes.
- **63 and 125 Hz in the hall.** EDT is 3.8 and 2.5 JNDs short and T20 1.7 and 3.2 JNDs
  long, as with the fitted set: the early part there is stronger than measured (see
  "The hall's early part at low frequencies" above). The hall's Schroeder frequency is
  about 45 Hz.

Fitting took 5 minutes for the seminar room (three rounds of tracing at the first
position's 10,000 rays per band) and 3.8 hours for the hall (four rounds), nearly all of
it the early stage's edge fade, which fewer rays do not shorten. Simulating again took
4 minutes and 83 minutes.

## Edge diffraction against the exact solution

`validation/edge_diffraction.py` compares the engine's edge diffraction with the
Biot-Tolstoy-Medwin (BTM) solution for rigid wedges (Svensson, Fred and Vanderkooy,
1999), which writes the diffracted sound as an integral along the edge: first order
over one edge, second order over two in turn, with half the integral when the two edges
share a face. The script integrates both along edges 80 m long, densest where the
shortest path touches them, and checks itself against the UTD for a half-plane, which
is exact far from the edge (within 0.03 dB). Levels are in dB re the direct sound in
free field over the same distance, the energy averaged over the engine's eight
frequencies in each octave band; sources and listeners are in metres from the edge.

Over a thin screen (first order), the engine's UTD is within 0.9 dB of BTM at 63 Hz
and 0.06 dB from 1 kHz up; Maekawa's chart reads up to 1.2 dB high:

| thin screen, engine - BTM (dB) | 63 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (-5, 0, -2) to (5, 0, -1), BTM -8.4 to -27.2 dB | -0.50 | -0.27 | -0.12 | -0.05 | -0.01 | -0.00 | +0.00 | +0.00 |
| (-2, 0, -0.2) to (6, 0, -3), BTM -6.9 to -24.6 dB | -0.77 | -0.46 | -0.24 | -0.11 | -0.04 | -0.01 | -0.00 | +0.00 |
| (-3, -1, -1) to (3, 2, -0.5), BTM -6.3 to -23.2 dB | -0.89 | -0.56 | -0.32 | -0.15 | -0.06 | -0.02 | -0.00 | +0.00 |

Over two edges (`second_order`): a block of width w with its top at z = 0, from
x = 0 to w, and two thin screens in a row. Each pair of positions is in line across the
edges and obliquely (the second from (-6, -2, -0.3) to (w + 2, 3, -4)):

| over two edges, engine - BTM (dB) | 63 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| block 0.3 m, (-3, 0, -1) to (3.3, 0, -1.5) | -1.72 | -1.25 | -0.67 | -0.14 | +0.19 | +0.30 | +0.24 | +0.14 |
| block 0.3 m, oblique | -0.65 | -0.30 | -0.19 | -0.16 | -0.13 | -0.09 | -0.07 | -0.07 |
| block 1 m, (-3, 0, -1) to (4, 0, -1.5) | -0.55 | -0.28 | -0.04 | +0.12 | +0.16 | +0.13 | +0.06 | +0.03 |
| block 1 m, oblique | -0.38 | -0.28 | -0.19 | -0.12 | -0.08 | -0.06 | -0.05 | -0.03 |
| block 3 m, (-3, 0, -1) to (6, 0, -1.5) | -0.20 | -0.07 | +0.02 | +0.05 | +0.05 | +0.03 | +0.01 | +0.01 |
| block 3 m, oblique | -0.28 | -0.19 | -0.13 | -0.08 | -0.06 | -0.04 | -0.02 | -0.03 |
| two screens 1 m apart | -1.86 | -1.23 | -0.74 | -0.37 | -0.12 | +0.00 | +0.01 | +0.02 |
| two screens 2 m apart, the second 0.4 m higher | -1.29 | -0.92 | -0.65 | -0.44 | -0.29 | -0.18 | -0.10 | -0.04 |

The levels themselves run from -9 dB at 63 Hz to -41 dB at 8 kHz. Adding its waves in
energy, the engine is low at low frequencies where the edges are close together, as over
a thin screen: the waves of an edge add in energy near their boundaries, where the exact
solution adds them with their phases. In the 63 and 125 Hz bands the engine adds them in
pressure by default (`simulation.early.coherent_max_band_hz`, see [Simulation](simulation.md#low-frequencies-early-reflections-in-pressure)), the four
terms of each coefficient and the paths' waves with their phases, and there it is within
0.05 dB of BTM over the thin screen and within 0.25 dB over two edges, but for the 0.3 m
block in line (0.6 dB):

| engine in pressure - BTM (dB) | 63 | 125 |
|---|---:|---:|
| thin screen, (-5, 0, -2) to (5, 0, -1) | +0.02 | +0.01 |
| thin screen, (-2, 0, -0.2) to (6, 0, -3) | +0.03 | +0.01 |
| thin screen, (-3, -1, -1) to (3, 2, -0.5) | +0.04 | +0.02 |
| block 0.3 m, in line | -0.61 | -0.50 |
| block 0.3 m, oblique | +0.12 | +0.24 |
| block 1 m, in line | +0.11 | +0.12 |
| block 1 m, oblique | +0.14 | +0.06 |
| block 3 m, in line | +0.23 | +0.16 |
| block 3 m, oblique | +0.08 | +0.04 |
| two screens 1 m apart | -0.11 | -0.12 |
| two screens 2 m apart, the second 0.4 m higher | +0.03 | -0.00 |

The textbook UTD for two edges, with the second edge's distance parameter for a
wave from a line, was up to 6 dB low there; the engine takes it from the curvature of
the first edge's wave where it reaches the second.

As the source rises through the plane of the 1 m block's top, 3 m in front of it, it
comes into sight of the far corner, and the path over that corner alone takes over
from the path over both; likewise as the listener rises behind the block. The largest
step between positions 1 mm apart, over ±1 cm and exactly in the plane, is 0.62 dB for
the source and 0.72 dB for the listener at 63 Hz with the waves added in energy, 0.01 dB
with them in pressure (the default in that band), and at most 0.07 dB from 1 kHz up.
With `first_order` the level drops from the path over one corner to nothing.

At the corner of a thin screen, the quarter of the plane x = 0 below y = 0 and z = 0,
the source at (-5, -3, -1) and the listener walking from (5, 2.5, -2) to (5, 4, -2)
1 cm at a time with `direct` and every band in pressure, the direct sound comes round
the screen's side edge at y = 3. The exact level, BTM over both edges, each from the
corner on (the sound from one edge to the other left out), changes smoothly there; the
engine's steps:

| a screen's corner, `direct` (dB) | 63 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| largest step, engine | 2.82 | 1.81 | 1.37 | 0.50 | 0.28 | 0.27 | 0.10 | 0.15 |
| largest step, BTM | 0.00 | 0.01 | 0.02 | 0.06 | 0.04 | 0.07 | 0.10 | 0.15 |
| largest difference, engine - BTM | 1.83 | 0.99 | 1.06 | 0.41 | 0.21 | 0.22 | 0.04 | 0.08 |

The path over the top edge still touches it 8 cm from the corner there, but its term is
dropped as the direct sound it borders appears: a term counts only while the specular
path is found on its lit side and missing on its dark side, and here the direct sound
appears round the end of the edge, not across its boundary. The UTD has no terms for
the corners of edges (see [Known issues](#known-issues)).

The early stage of a hall 30 x 20 x 10 m with four alcoves along each long wall
(16 vertical edges), with the source and the listener in alcoves out of each other's
sight, takes 55 ms per source and listener on four cores without diffraction (23
paths), 650 ms with `first_order` (995 paths) and 710 ms with `second_order` (1,010
paths), the medians of nine runs; on the same computer 4.0.0 takes about 1.0 and
1.1 s. The terms that make the other specular paths fade at the edges (next section)
take about 580 ms of that with `first_order` and give most of the paths, as each term
is a path over its edge, and so is each product of terms: without them the early stage
takes 70 ms with `first_order` (71 paths) and 135 ms with `second_order` (86 paths).

## Reflections that fade at edges against the exact solution

`validation/edge_fade.py` checks the terms that make every specular path fade in and out
at edges (see [Simulation](simulation.md#edge-diffraction)) where the exact solution is
known: a vertical edge standing on a rigid ground. The ground mirrors the edge into
itself, so the sound is that of an edge infinite in height from the source and from its
image in the ground: the geometric paths of each that reach the listener, plus each
one's BTM term (as above). The source stands 0.5 m up, and a listener 1.5 m up walks
across the edges' boundaries 1 cm at a time:

- the end of a thin screen, with `diffraction: direct`: the reflection off the ground
  crosses the shadow boundary with the direct sound, and only the fade bends it;
- the corner of a building (a right-angled wedge), with `direct` and with
  `first_order`: the listener passes the boundary of the reflection off one wall and
  then the corner's shadow boundary, each with its copy off the ground. With
  `first_order` the edge diffraction bends all of those paths itself, and the fade must
  add nothing to it;
- a strip 2 m wide (a thin screen on the ground), with `direct` and with `first_order`:
  the listener, on the source's side, walks through the reflection off the strip, which
  a face that narrow gives only in part at low frequencies;
- a gap of 4 m between two screens in line, with `direct`: behind one screen's end,
  through the gap and behind the other's.

For the strip and the gap the exact level adds each end's term to the geometric paths
and leaves out the sound diffracted from one end to the other (at the strip it changes
the level by at most 0.1 dB, at 63 to 250 Hz). Everything adds in pressure, as
outdoors, and the exact level is averaged over 96 frequencies in each octave band. The
largest difference from it along each line, in dB:

| largest difference from the exact level (dB) | 63 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| screen end, `direct`, with the fade | 0.04 | 0.02 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.00 |
| without | 5.90 | 5.79 | 5.29 | 3.78 | 5.24 | 4.43 | 3.71 | 3.72 |
| building corner, `direct`, with the fade | 0.08 | 0.05 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.04 |
| without | 5.93 | 5.85 | 5.52 | 4.11 | 5.21 | 4.48 | 3.64 | 3.52 |
| building corner, `first_order`, with the fade | 0.08 | 0.05 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.04 |
| without | 0.08 | 0.05 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.04 |
| 2 m strip, `direct`, with the fade | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.19 |
| without | 0.90 | 0.54 | 0.96 | 1.70 | 0.19 | 0.64 | 0.40 | 0.46 |
| 2 m strip, `first_order`, with the fade | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.19 |
| without | 0.41 | 0.27 | 0.29 | 0.13 | 0.02 | 0.02 | 0.02 | 0.19 |
| 4 m gap, `direct`, with the fade | 0.03 | 0.02 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 |
| without | 7.01 | 5.57 | 5.25 | 3.68 | 6.05 | 4.45 | 3.92 | 3.72 |

With the fade the level changes from one listener to the next as the exact level does:
the largest step along each line is the same as the exact level's to 0.01 dB in every
band, at most 0.20 dB at 8 kHz past the screen's end, but for 8 kHz on the strip, where
it is 0.06 dB and the exact level's 0.12 dB. Without it the level stepped by up to
8.6 dB past the screen's end, 7.6 dB past the corner and 9.2 dB past the gap's ends. At
the corner with `first_order` the fade changes the level by less than 0.001 dB: where
the edge diffraction already bends a path, the fade leaves out the terms it adds. On the
strip with `first_order` the edge diffraction bends the reflection off the strip, and
the fade adds its copy off the ground (0.4 dB at 63 Hz).

The edge diffraction's own tests outdoors found two more faults of the fade, now fixed
and tested (`tests/test_edge_fade.py`). Behind a barrier on a hard ground, the leg
before a reflection off the ground also took the barrier top's term, where the line to
the listener's image passes the barrier beyond the ground; that wave is the path over
the barrier with the reflection after it, which the edge diffraction has already, and
the level was up to 0.8 dB off at 125 Hz. And the end of a wall as high as a roof
elsewhere counted as standing on the roof's plane, so that right on the shadow line of
the wall's end a reflection off that plane from below came as strong as the sound
round the end: 2.8 to 6.3 dB too loud, and 3.1 dB too soft at 1 kHz, where the two
cancelled. Neither changes any of the levels above.

Behind a barrier 3 m high on a rigid ground (a thin screen across x = 0), with the
source at (-4, 0, 0.5) and a listener at x = 4 m rising from 1 to 8 m 1 cm at a time,
the exact level is that of the geometric paths and of the four paths over the top, from
the source and its image in the ground to the listener and its image, each by BTM. With
`first_order` every path over the top is the edge diffraction's own, and the fade
changes nothing (less than 0.001 dB):

| barrier on a rigid ground, `first_order` (dB) | 63 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| largest difference, `one_side` | 2.20 | 3.00 | 2.62 | 0.89 | 0.79 | 0.67 | 0.75 | 0.72 |
| largest difference, `both_sides` | 0.38 | 0.05 | 0.07 | 0.01 | 0.02 | 0.05 | 0.03 | 0.05 |
| largest step, `both_sides` | 0.35 | 0.14 | 0.12 | 0.05 | 0.09 | 0.15 | 0.20 | 0.16 |
| largest step, exact | 0.06 | 0.14 | 0.12 | 0.05 | 0.09 | 0.15 | 0.21 | 0.16 |

`one_side` leaves out the fourth path, reflected off the ground before the top and
again after it (see [Outdoor results](#outdoor-results-against-exact-solutions)).
With `both_sides` the level is within 0.07 dB of the exact level from 125 Hz up, and
steps like it but at 63 Hz: there it steps by 0.35 dB (`one_side` by 0.33 dB) at
z = 6.5 m, where the listener rises into sight of the reflection off the ground before
the barrier, and is 0.33 dB low on average above. The path over the top with the
reflection off the ground behind it is left out there (see
[Known issues](#known-issues)).

Behind a barrier 1 m thick on the same ground (two right-angled edges 1 m apart, 3 m
up), with the source at (-5, 0, 0.5), `second_order` and listeners 5 m behind it, the
exact level is BTM's sound over both edges from the source and its image to the
listener and its image (about 4 minutes a listener). The engine minus the exact level:

| thick barrier, `second_order` (dB) | 63 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| listener 0.5 m up | -9.85 | -6.96 | +3.03 | -5.66 | -5.53 | -5.17 | -5.07 | -5.41 |
| 1.5 m up | -5.05 | -0.27 | -0.48 | -3.21 | -2.43 | -2.13 | -2.17 | -2.41 |
| 2.5 m up | -1.09 | -5.48 | +3.84 | -2.54 | -2.10 | -1.88 | -1.74 | -1.90 |

A path over two edges takes no reflection off the ground, so of the four paths over
the barrier the engine has one: 1.7 to 5.7 dB low from 500 Hz up, and up to 9.9 dB low
and 3.8 dB high below, where the four interfere. `both_sides`, which pairs reflections
round one edge, and the fade change nothing here (see [Known issues](#known-issues)).

### Round a corner indoors

Indoors the paths that pass an edge reflect off the floor, the ceiling and the walls
several times, and their sounds add in energy above 125 Hz (each band's level is the
sum of its paths'). No exact solution covers a whole room, so the check is that the
level no longer jumps where a path appears or goes: the `gallery` example's visitor
walks away from the guide and round the corner at (8, 8) m, 1 cm at a time from y = 6
to 18 m along x = 3, 4 and 5.5 m (3,603 positions), with reflections up to third order
and `first_order`, as the example has them. The largest step between neighbours, in
dB:

| largest step (dB) | 63 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| with the fade | 0.06 | 0.10 | 0.02 | 0.03 | 0.04 | 0.05 | 0.07 | 0.13 |
| without | 3.03 | 3.72 | 2.38 | 2.52 | 2.64 | 2.73 | 2.78 | 2.83 |

Walking these lines found faults that made steps of 0.2 to 2.5 dB, now fixed: the
cross terms of the terms with their paths went to other paths, as the edge diffraction
hands the paths back sorted by arrival; of the two orders of a reflection off the floor
and off a wall (perpendicular planes, one image), one that cannot be was kept, and the
path over an edge may take the other order, so each path now takes the terms of both;
paths through the line where two such planes meet were dropped, found or not; a path
whose reflection off the floor lies just behind a wall was checked clear to that point
rather than to where the path over the corner reflects; a listener exactly in a face's
plane counted as in front of it; the paths of the highest order took the terms of
reflections off faces the listener is behind, though in front of those faces the early
stage follows no such reflection; the wave that takes over behind a face's plane from
a reflection the size scattering weakens (after Christensen and Rindel, as in ODEON,
from the third reflection of a path on with `first_order`) was not weakened alike, which made 0.72 dB at
63 Hz into the plane of the wall round the corner; and where the paths over the corner
reach its top, where it meets the ceiling, neither the leg before the reflection off
the ceiling nor the one after it took them. `test_no_step_round_the_gallery_corner`
walks seven of these places.

Steps of 0.16 to 0.30 dB at 63 and 125 Hz in three more places are gone too. At y =
12.00 and 17.34 m on x = 5.5 m several events stop a path that is not found, as its
reflection off the floor or the ceiling lies past its face where the corner blocks a
leg next to it, and such a path was left out, with its terms, though the path over the
corner, which reflects off the floor or the ceiling elsewhere, goes on: it is now
followed (see [Simulation](simulation.md#edge-diffraction)). At y = 16.40 m on x = 3 m
a path off wall_6, wall_7 and wall_2 reflects off wall_2 before wall_7 on one side and
after it on the other (perpendicular planes, one image), and the early stage's rays had
found only one of the two orders, which the fade took as the only one: it now tries
every order of reflections off perpendicular planes.

The largest steps left stand out from their neighbours' by at most 0.08 dB, all at 63
and 125 Hz; from 250 Hz up by at most 0.01 dB, where in 4.0.0 they stood out by up to
0.10 dB at 500 Hz to 1 kHz as the listener crossed the plane of wall_3 (y = 8 m). The
largest, 0.08 dB at 125 Hz (y = 13.33 m on x = 5.5 m), and one of 0.03 dB at 63 Hz
(9.81 m on x = 4 m) come from a path that the corner stops on one leg while another of
its reflection points slides past the end of its face into a corner of the room: the
path is left out there, though its path over the corner still reflects on the face. One
of 0.02 dB at 125 Hz (12.99 m on x = 3 m) is a leg that takes the corner's term on the
lit side on one side of the step and not on the other; the others, of up to 0.035 dB,
are not traced.

Where the edge diffraction already bends every path the early stage follows
(`max_order: 0` with `direct`, `max_order: 1` with `first_order`), the fade changes the
level along x = 4 m by less than 0.05 dB. It had taken away the terms of reflections
off faces the listener was behind, which the early stage follows nowhere at that order:
2.9 dB less at once at 63 Hz behind the plane of the wall round the corner.

### Along a street

The `street` example's car (see [A car passing by](demos.md#a-car-passing-by)) passes
the pavement, a balcony of the office block and the garden behind the noise barrier.
No exact solution covers a street either, so the check is again that the level does not
jump: the early level, A-weighted with the car's sound power, every 10 cm of the car's
125 m, and again 1 cm apart round each step of 10 cm that is among the four largest for
its listener in some band or in dB(A) (`validation/walks.py`). Outdoors paths
add in pressure in every band, so the level also swings, smoothly, where they go in
and out of step: in the garden by up to 3.2 dB(A) from one metre to the next, and 1 cm
apart by up to 0.24 dB(A) and 0.7 dB at 8 kHz, where a centimetre is a quarter of a
wavelength. The late part is left out, as its rays are drawn at random (the early
level repeats to the last bit): with the scene's rays few of them reach the garden,
and two runs of the same pass without a `seed` put the garden's level up to 2.3 dB(A)
apart at one place (x = -47 m; 0.45 dB(A) rms over the pass), the pavement's and the
balcony's up to 0.8 and 0.5 dB(A), each L_Amax within 0.02 dB.

On the pavement and the balcony the early level changes by at most 0.24 and 0.15 dB(A)
from one 10 cm to the next, and in the garden by at most 1.33 dB(A), where 1 kHz swings
by 2.5 dB at x = 28.8 m and does so smoothly 1 cm apart; without the fade the garden's
level stepped by up to 11.4 dB(A) from one metre to the next. The steps of 0.90 to 1.90
dB that two events stopping a path made in the garden, at x = -41.60, 30.41, 32.00 and
34.00 m (see the next paragraph), are gone: 1 cm apart, the level there now stands out
from its neighbours' by at most 0.11 dB in any band. The steps left that stand out by
more than 0.1 dB in some band 1 cm apart, but at 8 kHz, where a centimetre is a quarter
of a wavelength and the level swings by up to 0.8 dB from one to the next (three more
at 4 kHz in the garden, at x = 19.40, 28.72 and 28.88 m, are swings too: 1 mm apart
the level there changes smoothly, by at most 0.06 dB a millimetre), each the step from
1 cm before x to x, and how far it stands out in dB(A) and in the same band without the
fade (`validation/walks.py`):

| car at x (m) | listener | stands out (dB) | in | dB(A) | without the fade (dB) |
|---:|---|---:|---|---:|---:|
| -41.39 | garden | 1.13 | 1 kHz | 0.52 | 0.00 |
| -14.04 | garden | 1.41 | 63 Hz | 0.27 | 0.33 |
| -0.75 | garden | 1.29 | 63 Hz | 0.15 | 0.41 |
| -0.69 | garden | 1.37 | 63 Hz | 0.27 | 0.47 |
| 9.99 | garden | 0.82 | 63 Hz | 0.03 | 0.96 |
| 37.80 | pavement | 0.55 | 63 Hz | 0.03 | 1.09 |
| 58.79 | garden | 0.82 | 250 Hz | 0.09 | 0.93 |
| 59.09 | balcony | 0.29 | 63 Hz | 0.05 | 0.04 |
| 59.37 | balcony | 0.29 | 63 Hz | 0.07 | 0.15 |
| 60.03 | balcony | 0.40 | 63 Hz | 0.04 | 0.00 |
| 60.70 | pavement | 0.40 | 250 Hz | 0.00 | 0.00 |
| 60.80 | garden | 1.42 | 500 Hz | 0.32 | 0.29 |
| 60.87 | garden | 1.31 | 500 Hz | 0.19 | 0.26 |
| 61.02 | pavement | 0.10 | 63 Hz | 0.03 | 0.00 |
| 62.90 | pavement | 1.02 | 250 Hz | 0.23 | 0.61 |
| 63.10 | pavement | 1.18 | 250 Hz | 0.27 | 0.60 |
| 63.53 | balcony | 0.32 | 63 Hz | 0.00 | 0.18 |
| 66.53 | balcony | 0.16 | 1 kHz | 0.10 | 0.05 |
| 69.51 | balcony | 0.75 | 63 Hz | 0.09 | 0.70 |
| 69.61 | balcony | 0.73 | 63 Hz | 0.03 | 0.67 |

All but the one at 9.99 m are the corners of edges (see
[Known issues](#known-issues-of-the-edge-fade)): a path over the barrier's top, on its
own or with a reflection off the ground or the offices' façade before or after it,
reaches an end of the top (x = 14 or 44 m), or a path round the barrier's end at
x = 44 m reaches the end's top, where its term stops, the edge diffraction's own as
well as the fade's, whose term of the same path stops there with it and makes several
of them larger (at 60.80 m 0.29 dB without the fade, 1.42 dB with it). The largest in
dB(A), at -41.39 m, is the fade's alone: the term of a second-order reflection that the
barrier's top cuts off reaches the top's end at x = 14 m. The dip at 9.99 m is the next
paragraph's. `tests/test_walks.py` walks the garden round the four largest steps in CI
and holds them to 1.5 dB in a band and 0.6 dB(A).

The four steps that are gone came from paths that two events stop at once. At x =
-41.60, 30.41 and 32.00 m the reflection off the offices' façade of a path that the
houses' corner or the barrier blocks reaches an end of the façade (x = -30 and 30 m):
the path went from being stopped by two events to one, or back, and its path over the
edge came or went at once (the edge diffraction's own path over the barrier with that
reflection does the same at 30.41 m, 0.60 dB at 125 Hz without the fade). At 34.00 m a
second-order path off the façade and the houses' end wall, stopped where it misses the
wall, has its last leg pass the barrier's end as well. The fade now follows such paths
(see [Simulation](simulation.md#edge-diffraction)).

At x = 10.0 m, where the car crosses the plane of the houses' end wall, the sound over
the houses' corner dips by up to 10 dB at 63 Hz within 2 cm (1.0 dB in the garden's
level over 1 cm). It is the edge diffraction's own term, with or without the fade: it
gives the wall's reflection the factor of a plane wave, which for the brick's impedance
turns to -1 at grazing incidence, so that the reflection cancels the incident sound
there. With a rigid wall the dip is gone. The reflection off the ground takes the
factor of a spherical wave instead (see
[Ground reflection](#ground-reflection-against-the-exact-solution)); whether that factor
would also be the better one here is not checked.

### Known issues of the edge fade

- **Paths that several events stop.** A path that is not found is followed where one
  edge explains each run of the events that stop it, or two edges explain them between
  them, in at most two runs. Elsewhere it is left out with its terms, as where one of
  its reflection points slides into a corner of the room while the path over the edge
  still reflects on the face, which makes the steps of up to 0.08 dB left round the
  gallery's corner (see above). The edge diffraction's own paths over an edge with a
  reflection before or after it stop where that reflection leaves its face, which the
  fade makes up for where it follows the path; without the fade the street's garden
  steps by 0.60 dB at 125 Hz there.
- **Corners of edges.** A term stops where the point it diffracts at leaves the end of
  its edge, as the UTD has no terms for the corners of edges, unless that end lies in a
  plane the path reflects off next to the event, where the edge goes on as its image in
  the plane (for a leg that stops a path, the image's term is found on the leg on the
  other side of the reflection); the edge diffraction's own paths stop there alike.
  Along the street the paths over the barrier's top make steps of up to 1.42 dB in one
  band and 0.52 dB(A) where they reach its ends (see above), the edge diffraction's own
  up to 1.09 dB without the fade. Past the corner of a thin screen the edge diffraction
  alone steps by 2.8 dB at 63 Hz (see
  [Edge diffraction against the exact solution](#edge-diffraction-against-the-exact-solution)).
  A term weighted by the part of its Fresnel zone that the edge covers, as BTM's
  integral over a finite edge has it, and kept where its path appears round the edge's
  end, would remove these steps; it is an open item on the [roadmap](roadmap.md).
  `tests/test_walks.py` holds the garden's four largest steps to 1.5 dB in a band and
  0.6 dB(A).
- **Terms that cancel, in the bands that add in energy.** Indoors paths add in energy
  from 250 Hz up (`coherent_max_band_hz`). Where both edges of a narrow face border its
  reflection, their terms nearly cancel there at low frequencies; added in energy they
  would add up instead. The terms are therefore kept as waves (`engine.clusters` in the
  [API reference](api.md#edge-diffraction)): on the lit side of its boundary a term
  continues its geometric wave as far as it is coherent with it over the band, as it
  arrives later by its detour, and on the dark side, where that wave is gone, by the
  UTD's share, so that deep in a shadow all the terms of an edge are one wave; the rest
  of the term is part of its event's diffracted wave, with the terms of the event's
  other edges and the edge diffraction's paths there. The products of a path's terms
  split term by term, the events the path gets past included. A wave's energy is that
  of its pieces added up in pressure, averaged over each band's frequencies and the
  slices they stand for, never below nothing; the wave of a found path is the path's,
  and where its pieces' cross terms take more than it has, the rest comes off them in
  proportion. Walking the `archway`'s listener behind the wall 1 cm at a time
  (x = 7.6 m, y = 5.3 to 7.8 m), where the reflection off the arch's jamb, 40 cm deep,
  appears, the level steps by at most 0.08 dB from 250 Hz to 1 kHz, between y = 5.39
  and 5.40 m (see the next item), and by at most 0.16 dB from 2 kHz up. In 4.0.0, where
  the cross terms between pairs of terms could take more than the terms carried and the
  sum stopped at zero, it stepped by 1.37 dB at 500 Hz and 1.06 dB at 250 Hz at
  y = 6.14 m and by 0.92 dB at 250 Hz at 7.33 m, and by up to 4.4 dB before the terms
  of one event took their cross terms with each other. The edge diffraction's own
  paths, without the fade, step there by up to 1.56 dB at 63 Hz and 1.88 dB from 250 Hz
  up. Averaged over the walk, the level is 0.4, 1.1 and 0.7 dB above that with every
  band in pressure (`coherent_max_band_hz: 8000`) at 250, 500 and 1000 Hz (4.0.0: 0.1,
  1.6 and 0.9 dB) and 0.1 to 0.4 dB below it from 2 kHz up: not traced.
- **Cut off by a thick end.** Behind the `archway`'s wall the level steps by 0.57 dB at
  63 Hz and 0.41 dB at 125 Hz between y = 5.39 and 5.40 m, and by 0.08 dB or less from
  250 Hz up, as it does with every band in pressure. Reflections that the end of the
  arch's jamb cuts off are followed while the leg they are cut off on hits the end's
  face, and left out once it hits the jamb's face on the far side, 40 cm on, which is a
  face of the end's other corner and not of the edge whose shadow they are in. Letting
  that other corner explain them removed the step but made one of 1.75 dB at y = 5.90 m,
  where a reflection point of one of those paths also leaves its face and the path,
  stopped at two events, is left out at once; so it is not done. The other steps along
  the walk, of up to 0.24 dB at 63 and 125 Hz and 0.07 dB from 250 Hz up, are there in
  pressure too; at y = 6.83 to 6.84 m and 7.00 to 7.01 m, terms of cut-off second- and
  third-order reflections come or go, and why is not traced. The fix would take a term
  over both corners of the end in turn, as `second_order` has for the direct sound,
  which the fade does not have (an open item on the [roadmap](roadmap.md)).
  `tests/test_walks.py` walks the whole line in CI and holds the steps to 0.65 dB at
  63 Hz, 0.5 dB at 125 Hz, 0.1 dB from 250 Hz to 2 kHz and 0.2 dB above.
- **Products of terms.** The terms of a path's events multiply, as the path must get
  past each in turn, an approximation for terms at different edges. Each product is a
  path that arrives later by the detours of all its terms, an approximation too for
  terms bent far from their boundaries; products below -60 dB of their path are left
  out in the bands where paths add in pressure. Where two events meet the same edge, the
  product is a wave bent twice by one edge, which the exact solution does not have. It
  is not checked on its own; behind the corner of the tests' L-shaped room such
  products come 23 dB below the paths over the corner.
- **Order of reflections.** Where the edge diffraction does not bend it, a path of the
  highest order the early stage follows takes no terms for the reflections off the
  edge's faces, behind their planes or in front, as the early stage follows no path
  with one more reflection: near an edge the level is that of the order chosen, as for
  the specular paths.
- **What is checked.** Against exact solutions: vertical edges standing on a rigid
  ground (a screen's end, a building's corner, a 2 m strip and a 4 m gap) with
  `direct`, and the corner and the strip with `first_order` too; a barrier's top on a
  rigid ground, with a reflection off the ground before or after it, with
  `first_order` and with `both_sides`; and a thick barrier with `second_order`, whose
  paths over two edges take no terms of the fade. Only for steps, with no exact level:
  paths of two or more reflections indoors (the gallery, and with `second_order`
  behind the archway) and in a street, walked in CI (`tests/test_walks.py`) and in full
  by `validation/walks.py`. Not checked: surfaces with an impedance against an exact
  solution. Nothing is checked against measurements.

## Ground reflection against the exact solution

`validation/ground_reflection.py` compares the reflection off a porous ground (see
[Outdoor scenes](outdoor.md#ground-impedance)) with the exact solution for a point
source over a locally reacting plane, Sommerfeld's integral as Chien and Soroka (1975)
write it, which the script integrates numerically. The engine reflects with the
Weyl-van der Pol factor Q, the integral's asymptotic form. Levels are those of the
direct sound and its reflection together, in dB over the direct sound in free field.
The grounds follow Miki's model with Nord2000's flow resistivities; the source and the
listener are 2 and 1.2 m high 5 m apart, and 1 and 1.5 m high 20 and 100 m apart.

Q against the exact solution, the largest difference in level at the third-octave
centres from 50 Hz to 5 kHz (in brackets, in dips more than 20 dB deep):

| ground (flow resistivity, kPa s/m²) | 5 m | 20 m | 100 m |
|---|---:|---:|---:|
| snow (12.5) | 0.16 | 0.19 | 0.12 |
| snow 10 cm deep on ice (12.5) | 0.25 | 0.36 (2.2) | 0.41 (0.8) |
| grass (200) | 0.02 | 0.04 | 0.04 |
| gravel (2000) | 0.00 | 0.01 | 0.01 |
| asphalt (20000) | 0.00 | 0.00 | 0.00 |

Q is furthest off over the thin layer of snow, the most reactive of these grounds, and
by more than half a decibel only in its deepest dips.

**Extended reaction.** `--extended` checks grounds whose impedance changes with the
angle of incidence (`impedance.reaction: extended`, see
[Ground impedance](outdoor.md#ground-impedance)): sound also travels sideways inside
them, and a plane wave meeting them at the angle θ from the normal meets the impedance
Zc n / sqrt(n² - sin² θ), n the ground's complex index, and for a layer on a hard
backing that times coth(j k d sqrt(n² - sin² θ)). The exact field of a point source over
such a ground is the Sommerfeld integral with that reflection at every horizontal
wavenumber, which the fast field program (`validation/ffp.py`, exact in still air, see
[Outdoor sound in wind against a fast field program](#outdoor-sound-in-wind-against-a-fast-field-program))
integrates. The engine reflects with Q taking the impedance at the reflection's own
angle; the table gives the largest difference in level from the exact field at the
third-octave centres from 50 Hz to 2 kHz, outside dips more than 20 dB deep, for that
and for Q with the locally reacting impedance, and how much the extended reaction tells
(the exact fields of the two reactions apart), the source 1 m and the listener 1.5 m
high:

| ground (flow resistivity, kPa s/m²) | 20 m: Q at the angle | Q local | reaction tells | 100 m: Q at the angle | Q local | reaction tells |
|---|---:|---:|---:|---:|---:|---:|
| fresh snow (5) | 0.44 | 0.94 | 0.96 | 0.11 | 0.65 | 0.66 |
| snow (12.5) | 0.22 | 0.59 | 0.59 | 0.13 | 0.39 | 0.38 |
| snow 10 cm deep on ice (12.5) | 1.46 | 6.36 | 3.74 | 0.37 | 2.05 | 2.45 |
| forest floor (31.5) | 0.14 | 0.35 | 0.35 | 0.09 | 0.29 | 0.29 |
| grass (200) | 0.06 | 0.09 | 0.10 | 0.04 | 0.09 | 0.11 |

The extended reaction tells by a tenth of a decibel over grass, by a third to one
decibel over snow and forest floors and by 2.5 to 4 dB over the thin layer of snow,
and Q with the impedance at the angle follows it to within half a decibel but over
the layer, where it still misses the exact field by 1.5 dB at 20 m: Q is the
locally reacting plane's spherical wave with the impedance swapped in, and the ground
wave over a layer of extended reaction is not quite that (Li, Attenborough and Heap
1998). The parabolic equation, which takes the impedance at grazing incidence,
against the same exact field at 63 Hz to 2 kHz:

| ground | 100 m | 400 m |
|---|---:|---:|
| fresh snow (5) | 0.06 | 0.05 |
| snow (12.5) | 0.35 | 0.05 |
| snow 10 cm deep on ice (12.5) | 0.84 | 1.05 |
| forest floor (31.5) | 0.37 | 0.27 |
| grass (200) | 0.12 | 0.16 |

The engine's early part over grass (image sources to first order, the direct sound and
its reflection added in pressure, as outdoors by default) against the exact solution,
whose energy is averaged over each octave band, as the engine reads a band:

| over grass, exact (dB) | 63 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 m, heights 2 and 1.2 m | 3.46 | -2.07 | 1.21 | 1.10 | 0.89 | 0.80 | 0.24 | 0.25 |
| 20 m, heights 1 and 1.5 m | 5.46 | 4.00 | -0.49 | -5.28 | 2.46 | 0.19 | 1.65 | 1.93 |
| 100 m, heights 1 and 1.5 m | 5.49 | 2.75 | -6.86 | -16.69 | -7.22 | -0.15 | 4.61 | 2.93 |

| over grass, engine - exact (dB) | 63 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 m, heights 2 and 1.2 m | -0.01 | +0.00 | -0.00 | -0.01 | -0.01 | -0.01 | +0.01 | -0.03 |
| 20 m, heights 1 and 1.5 m | -0.01 | -0.02 | -0.03 | +0.01 | +0.00 | -0.00 | -0.00 | -0.01 |
| 100 m, heights 1 and 1.5 m | -0.01 | -0.01 | -0.07 | +0.01 | +0.00 | +0.00 | -0.00 | -0.00 |

The ground dip near 500 Hz deepens with distance, as the reflection comes in nearer
grazing. Read through the analysis filters, whose skirts reach into the neighbouring
bands, a sharp dip fills in by up to about 1 dB more than the band averages show.

The parabolic equation that gives the sound over the open ground in wind and
temperature gradients (see [Outdoor scenes](outdoor.md#wind-and-temperature)), here in
still air against the exact solution (`--parabolic`): the largest difference in level
at the third-octave centres from 63 Hz to 2 kHz, from a source 1 m high to a listener
1.5 m high, and from a source 119 m high (a wind turbine's hub) to one 1.7 m high:

| ground | 25 m | 100 m | 400 m | 178 m, from 119 m | 535 m | 845 m |
|---|---:|---:|---:|---:|---:|---:|
| snow (12.5) | 0.64 | 0.38 | 0.39 | 0.26 | 0.02 | 0.04 |
| snow 10 cm deep on ice (12.5) | 1.81 | 0.97 | 1.05 | 0.41 | 0.06 | 0.08 |
| grass (200) | 0.13 | 0.14 | 0.16 | 0.28 | 0.02 | 0.05 |
| gravel (2000) | 0.12 | 0.15 | 0.19 | 0.45 | 0.02 | 0.10 |
| asphalt (20000) | 0.17 | 0.05 | 0.09 | 0.26 | 0.04 | 0.02 |

It is furthest off over the softest grounds near a low source at low frequencies (the
1.81 dB in a dip 36 dB deep at 160 Hz), where the field it starts from, which stands
for the source and its image, matters most. From the hub, 178 m away comes in 33
degrees from the horizontal, about as steep as it is used (34 degrees); in the engine
the paths come in flatter there, and rays take the steeper ones.

The wave solver fits the ground's admittance with branches without mass (see
[Simulation](simulation.md#low-frequencies-the-wave-solver)). Over grass, against a
rigid ground, with the source and the listener 1 and 1.5 m high 10 m apart, from the
first 25 ms after the reflection, before anything off the grid's edges arrives (the
solver up to its 250 Hz band):

| Hz | 40 | 63 | 80 | 100 | 125 | 160 | 200 | 250 | 315 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact (dB) | -0.17 | -0.43 | -0.66 | -0.98 | -1.44 | -2.20 | -3.21 | -4.73 | -7.11 |
| wave - exact (dB) | +0.02 | -0.02 | +0.01 | +0.01 | +0.02 | +0.02 | +0.05 | +0.10 | +0.11 |

## Outdoor propagation against measurements (Hannover)

The Hannover wind turbine propagation data (Könecke, Hörmeyer, Bohne and Rolfes, 2021,
[DOI 10.25835/0012136](https://doi.org/10.25835/0012136), CC BY-SA 3.0) measured a wind
turbine 178, 535 and 845 m away, with microphones 1.7 m above a field on one line, in
ten spells of five minutes in different weather, and gives the loss from the nearest
microphone to the two others in the third-octave bands from 80 Hz to 2 kHz, with the
inputs and the result of the authors' parabolic equation model.
`validation/outdoor_propagation.py` downloads the ten cases and builds each as the
authors modelled it: three point sources of equal power 71, 119 and 167 m up across the
rotor, a flat ground of Miki's impedance (200 kPa s/m²), the air measured 53 m up, and
their effective speed of sound c(z) = a0 + alog ln(1 + z / z0) + alin z as a measured
atmosphere, the log term as the wind and the linear one as the temperature. The scene
runs in third-octave bands with this package's defaults outdoors, the three sources'
energies added at each microphone.

Overall is the energy sum of the fifteen band losses, as the data gives it (so it reads
about 15 dB above any one band); then the mean absolute difference from the measured
loss over the bands. Each pair is at 535 m / 845 m, in dB:

| case | weather | overall, measured | published model | this package | mean difference, published model | this package |
|---|---|---|---|---|---|---|
| 1 | light downwind | 22.6 / 27.5 | 21.6 / 26.6 | 23.3 / 28.9 | 1.8 / 3.2 | 1.7 / 2.3 |
| 2 | light downwind | 22.2 / 26.7 | 21.6 / 26.8 | 23.3 / 28.9 | 1.6 / 2.8 | 1.9 / 2.7 |
| 3 | cross/downwind | 20.8 / 26.5 | 21.4 / 26.8 | 23.1 / 28.7 | 1.8 / 2.7 | 2.2 / 2.7 |
| 4 | cross/downwind | 20.6 / 26.3 | 21.3 / 26.9 | 23.1 / 28.6 | 2.0 / 2.7 | 2.3 / 2.6 |
| 5 | cross/upwind | 21.0 / 26.6 | 21.0 / 25.9 | 23.0 / 28.3 | 1.9 / 2.5 | 2.0 / 2.3 |
| 6 | cross/upwind | 20.6 / 25.7 | 20.9 / 26.0 | 22.9 / 28.2 | 1.7 / 2.8 | 2.4 / 2.7 |
| 7 | light upwind | 21.9 / 26.2 | 20.7 / 24.9 | 22.7 / 27.7 | 2.0 / 2.2 | 2.2 / 2.2 |
| 8 | light upwind | 22.2 / 27.0 | 20.5 / 25.1 | 22.7 / 27.9 | 2.2 / 2.2 | 2.3 / 2.1 |
| 9 | strong upwind | 23.1 / 27.4 | 19.9 / 52.3 | 22.0 / 27.8 | 3.2 / 8.3 | 2.0 / 2.6 |
| 10 | strong upwind | 23.1 / 27.2 | 19.9 / 50.0 | 22.1 / 27.9 | 3.1 / 7.7 | 2.0 / 2.5 |

In the eight cases of light winds, this package is about as close to the measurements
as the authors' model: 1.7 to 2.4 dB per band on average at 535 m, 2.1 to 2.7 dB at 845
m, against their 1.6 to 2.2 and 2.2 to 3.2 dB; the spread of the measurements over the
five minutes is 1 to 3 dB. Both lose a few decibels more than was measured from 80 to
160 Hz, this package more (overall 1 to 2.5 dB more loss at 845 m). These winds bend
the sound little over these distances: in still air (`--still`) the overall losses
move by 1.1 dB at most and the mean differences by 0.4 dB at most, except downwind at
845 m in cases 1 and 2, which still air matches better (1.4 and 1.8 dB): there the wind
takes 1 to 3 dB more from 100 to 250 Hz than was measured, the dip of the ground
moving down in frequency as the sound comes in steeper. The field is flat and smooth,
as the authors modelled it, so its material scatters nothing: with the few late rays
that get this far, random scattering would only add noise (up to 1 dB in a band).

Cases 9 and 10 are strong upwind. The published model puts 845 m deep in a shadow,
50 dB overall where 27 dB was measured, and the authors put the gap down to
turbulence, which scatters sound into a shadow and which neither model has. Here the
rays in the same profile reach 845 m from the two upper sources: the shadow starts 690
m away for the source 71 m up, but 980 and 1230 m away for those 119 and 167 m up, so
only the lowest one is in it, and the loss stays within about 1 dB of the measured
one overall.

The data gives no measure of the eddies. With `--turbulence moderate` the overall
losses drop by up to 0.4 dB at 845 m and 0.2 dB at 535 m, as the eddies fill the ground
dip a little; the mean differences move by 0.3 dB at most, better at 845 m in eight
cases and 0.2 dB worse in case 8. In cases 9 and 10 the overall loss at 845 m comes to
27.5 dB, against 27.4 and 27.2 measured.

## Turbulence against a parabolic equation through random eddies

`validation/turbulence.py` marches the engine's own parabolic equation (checked against
the exact solution, see
[Ground reflection](#ground-reflection-against-the-exact-solution)) through random
eddies and averages its sound's energy over 16 of them (Gilbert, Raspet and Di 1990):
each is a field of the index of refraction drawn with the von Kármán spectrum, outer
scale 10 m, in the vertical plane through the source and the listener. Both checks
fitted constants of the engine, so they show how well those fits hold over the cases
rather than test it independently.

**The ground dip.** In still air with the `strong` eddies, the level of the direct sound
and its reflection off the ground, 50, 100, 200 and 400 m away at 13 frequencies from
125 Hz to 2 kHz, against the parabolic equation's: the engine's coherence factor, and the
Markov approximation's alone (Clifford and Lataitis 1983), which fills the dip over soft
grounds far more than the parabolic equation does. Mean (largest) absolute difference,
in dB:

| ground | source / listener (m) | engine | Markov alone |
|---|---|---:|---:|
| grass | 2 / 1.5 | 0.75 (3.2) | 2.57 (13.6) |
| grass | 5 / 1.5 | 0.64 (3.8) | 1.61 (7.1) |
| loose soil | 1 / 1 | 1.29 (4.6) | 3.55 (14.1) |
| gravel | 2 / 1.5 | 1.27 (7.0) | 2.06 (11.7) |
| grass | 10 / 2 | 0.58 (2.1) | 0.93 (5.8) |
| asphalt | 10 / 2 | 0.92 (3.5) | 0.86 (3.7) |
| asphalt | 5 / 1.5 | 0.93 (7.2) | 0.91 (7.7) |
| compacted soil | 3 / 3 | 0.75 (3.5) | 1.49 (7.8) |

Over the five soft grounds the engine is 0.9 dB off on average, against 2.1 for the
Markov approximation alone; over the three hard ones (asphalt and compacted soil) 0.9
against 1.1.

**Into a shadow.** Over grass, the source 2 m and the listener 1.5 m up, on a sunny day
and upwind in the `windy` preset, deep in the shadow (2.5 to 9 times as far as it
starts), from 250 Hz to 2 kHz: the parabolic equation's level through the eddies against
the engine's, which is the parabolic equation's in still air plus what the eddies
scatter into the shadow. Mean (largest) absolute difference:

| atmosphere | shadow from | `moderate` eddies | `strong` eddies |
|---|---:|---:|---:|
| sunny day | 56 m | 0.8 dB (2.9) | 1.2 dB (3.5) |
| windy, upwind | 31 m | 0.7 dB (2.8) | 0.9 dB (2.5) |

The farthest points with `moderate` eddies, in dB over the free field:

| Hz | sunny day 503 m, still air | through eddies | engine | windy 281 m, still air | through eddies | engine |
|---|---:|---:|---:|---:|---:|---:|
| 250 | -33.3 | -27.1 | -25.9 | -25.8 | -25.7 | -24.8 |
| 500 | -49.2 | -20.4 | -21.8 | -61.1 | -26.3 | -24.8 |
| 1000 | -68.2 | -13.4 | -14.1 | -64.9 | -24.3 | -21.5 |
| 2000 | -61.9 | -11.0 | -11.6 | -58.4 | -19.7 | -21.1 |

At these points, without the eddies the shadow sinks 49 to 68 dB below the free field
from 500 Hz; through them it stays 11 to 26 dB below, and the engine follows within 3 dB.
Only these heights and these two profiles were checked.

**Eddies weaker with height.** `--profile` runs both checks with the structure
parameters given 2 m up (`height_m`, see [Turbulence](outdoor.md#turbulence)) and
falling as z^(-2/3): each field is scaled in height by the square root of that, rising
right down to the roughness length 5 cm above the ground, where it is 12 times as
strong as at 2 m, while the engine takes Cn² no nearer the ground than the eddies'
outer scale, 10 m (0.34 of the value at 2 m). The ground dip, mean (largest) absolute
difference in dB, with a third column for the engine taking Cn² at each pair's mean
height right down to the ground instead:

| ground | source / listener (m) | engine | Markov alone | engine, Cn² down to the ground |
|---|---|---:|---:|---:|
| grass | 2 / 1.5 | 1.26 (8.0) | 2.56 (12.2) | 2.34 (9.9) |
| grass | 5 / 1.5 | 0.89 (5.9) | 1.38 (6.4) | 1.15 (6.1) |
| loose soil | 1 / 1 | 0.90 (4.9) | 3.18 (14.3) | 2.50 (9.1) |
| gravel | 2 / 1.5 | 1.79 (8.3) | 2.16 (9.4) | 2.49 (10.5) |
| grass | 10 / 2 | 0.71 (4.2) | 0.81 (3.8) | 0.62 (2.9) |
| asphalt | 10 / 2 | 1.58 (11.3) | 1.38 (8.7) | 1.17 (9.2) |
| asphalt | 5 / 1.5 | 1.60 (10.3) | 1.52 (8.6) | 1.45 (7.0) |
| compacted soil | 3 / 3 | 1.10 (4.0) | 1.48 (8.0) | 1.37 (6.7) |

Over the five soft grounds the engine is 1.1 dB off on average (0.9 with eddies
the same at every height, 1.8 taking Cn² down to the ground), over the three
hard ones 1.4 (0.9 and 1.3). The field's thin layer of stronger
eddies near the ground hardly tells in the parabolic equation: fields scaled only above
2 m and fields scaled only below it fill the dip within 2 dB of the fields weakened
above and strengthened below, and of the same fields the same at every height,
respectively, because the waves average over the layer, which is thinner than their
Fresnel zone, and both the direct sound and its reflection cross it alike; what parts
them is the eddies over the Fresnel zone's height, most of it above the paths. Cn² at
the greater of a pair's mean height and the Fresnel zone's half-width, sqrt(lambda L /
4), does as well (1.1 dB over all the cases against 1.2 at the outer scale); the engine
keeps the outer scale, one rule for the coherence, the scattering and the shaking alike.
Into the shadows, with the same profile:

| atmosphere | shadow from | `moderate` eddies | `strong` eddies |
|---|---:|---:|---:|
| sunny day | 56 m | 1.3 dB (4.2) | 1.7 dB (5.8) |
| windy, upwind | 31 m | 1.9 dB (8.7) | 2.9 dB (12.2) |

Deep in the shadow the eddies that scatter and shake sound in are those over the
limiting rays, tens of metres up, where the profile leaves a third to a seventh of the
strength at 2 m: the parabolic equation's level through these eddies is 4 to 10 dB
(sunny day) and up to 6 dB (upwind) below that through eddies the same at every height
at the farthest points from 500 Hz, and the engine follows within 1.3 to 2.9 dB on
average (2.8 to 3.3 with Cn² at the geometric mean of the source's and the listener's
heights right down to the ground). Its largest misses are at the nearest points upwind
at 2 kHz, where the field's strong layer at the ground brings the parabolic equation
back within 4 dB of the free field with the `strong` eddies (11 with eddies the same at
every height) while the engine, stopping at the outer scale, stays 16 below: near the
shadow's edge the shaking does feel the eddies right at the ground, and the engine is
low there.

## Vegetation against a particle simulation

`validation/vegetation.py` checks the late rays through vegetation against a particle
simulation written apart from the engine, in numpy: a source in the middle of a box
200 m across whose walls absorb everything, a slab of plants from x = 10 to 40 m across
it, and three receivers, spheres 10 m in radius 50 m from the source, through the slab
(+x), beside it (+y) and behind the source (-x). The particles go straight to their
next scattering, drawn at the medium's rate, and leave in a direction drawn from its
phase function. Both count the energy in the spheres by the length of track through
them, and as the engine's rain does: next-event estimates from every scattering in
foliage, and from every scattering off stems the track the next leg would have through
each sphere, averaged over the turns. Each run also sends the same particles through
empty air, and the table gives its energy less that run's plus the exact free field's,
so the direct sound's own randomness drops out. Energy over the free field's at 50 m,
mean ± standard error of 16 runs of a million particles:

| medium | receiver | reference, spheres | engine, spheres | reference, rain | engine, rain |
|---|---|---|---|---|---|
| none (exact 1.008) | through | 1.008 ± 0.002 | 1.010 ± 0.002 | - | - |
| none (exact 1.008) | beside | 1.003 ± 0.003 | 1.011 ± 0.003 | - | - |
| none (exact 1.008) | behind | 1.009 ± 0.002 | 1.006 ± 0.002 | - | - |
| isotropic | through | 0.997 ± 0.002 | 0.995 ± 0.003 | 0.990 ± 0.002 | 0.988 ± 0.002 |
| isotropic | beside | 1.418 ± 0.002 | 1.419 ± 0.002 | 1.418 ± 0.000 | 1.418 ± 0.000 |
| isotropic | behind | 1.204 ± 0.002 | 1.202 ± 0.001 | 1.202 ± 0.000 | 1.202 ± 0.000 |
| forward | through | 0.737 ± 0.002 | 0.734 ± 0.003 | 0.730 ± 0.001 | 0.732 ± 0.002 |
| forward | beside | 1.103 ± 0.001 | 1.102 ± 0.001 | 1.098 ± 0.000 | 1.098 ± 0.000 |
| forward | behind | 1.030 ± 0.000 | 1.030 ± 0.000 | 1.030 ± 0.000 | 1.030 ± 0.000 |
| stems | through | 0.685 ± 0.003 | 0.682 ± 0.002 | 0.685 ± 0.002 | 0.682 ± 0.001 |
| stems | beside | 1.254 ± 0.001 | 1.253 ± 0.001 | 1.254 ± 0.001 | 1.254 ± 0.001 |
| stems | behind | 1.187 ± 0.001 | 1.185 ± 0.001 | 1.186 ± 0.000 | 1.185 ± 0.000 |
| leaves | through | 0.947 ± 0.003 | 0.949 ± 0.002 | 0.939 ± 0.002 | 0.943 ± 0.002 |
| leaves | beside | 1.529 ± 0.002 | 1.525 ± 0.002 | 1.528 ± 0.001 | 1.527 ± 0.001 |
| leaves | behind | 1.252 ± 0.001 | 1.252 ± 0.001 | 1.251 ± 0.000 | 1.252 ± 0.000 |
| two bands, band 1 | through | 0.994 ± 0.003 | 1.004 ± 0.004 | 0.988 ± 0.002 | 0.991 ± 0.003 |
| two bands, band 1 | beside | 1.418 ± 0.002 | 1.421 ± 0.002 | 1.417 ± 0.000 | 1.417 ± 0.000 |
| two bands, band 1 | behind | 1.203 ± 0.001 | 1.203 ± 0.002 | 1.202 ± 0.000 | 1.202 ± 0.000 |
| two bands, band 2 | through | 0.736 ± 0.002 | 0.743 ± 0.003 | 0.732 ± 0.001 | 0.734 ± 0.003 |
| two bands, band 2 | beside | 1.101 ± 0.001 | 1.103 ± 0.001 | 1.098 ± 0.000 | 1.098 ± 0.000 |
| two bands, band 2 | behind | 1.030 ± 0.000 | 1.030 ± 0.000 | 1.030 ± 0.000 | 1.030 ± 0.000 |

The five media: foliage that scatters every way alike, 0.03 per metre; foliage that
scatters forwards (mean cosine 0.6), 0.02 per metre, and absorbs 0.01; stems, 0.03 per
metre of horizontal travel, turning sound by a mean cosine of -0.3; leaves of 0.25 m²
per m³, 5 cm wide, as the engine works them out in the 4 kHz band
([Leaves](#leaves-against-closed-forms-and-iso-9613-2): 0.0535 per metre scattered
with a mean cosine of 0.14, 0.00075 absorbed); and the first two in two bands at
once, which the engine traces with one set of rays. The engine agrees with the
particles within 1 % through the slab (0.4 % with one medium alone) and 0.3 % beside
it and behind the source, with the spheres and with the rain: at most two to three of
their standard errors, and the particles' own runs differ as much from one set of
seeds to another (0.990 and 0.988 through the slab, rain, the same foliage alone and
in two bands). Next to the slab
the rain gives a little less than the spheres in both programs alike, as it takes
the scattered sound at the sphere's centre and the spheres its mean over the sphere.
Beside and behind the slab the rain is steadier than the spheres. Only a slab was
checked, and against a simulation of the same model, not against measurements
through trees.

## Leaves against closed forms and ISO 9613-2

`validation/leaves.py` checks foliage given as leaves (`foliage: {leaf_area_per_m3,
leaf_width}`, see [Vegetation](outdoor.md#vegetation)), which the engine takes as thin
rigid discs turned every way alike. The disc's scattering cross-section over its area
against its closed forms: small against the wavelength, the dipole of the air the
disc holds back, 16 (kr)⁴ / (81 π²) (from a disc's added mass, 8 ρ r³ / 3, a third of
it turned every way alike); large, twice its mean projected area. In between the
engine bridges the two, as the exact cross-section of a disc about a wavelength
across is not in closed form:

| kr | 0.1 | 0.3 | 1 | 2 | 3 | 5 | 10 | 30 |
|---|---|---|---|---|---|---|---|---|
| dipole | 2.0e-6 | 1.6e-4 | 0.020 | 0.32 | 1.6 | 13 | 200 | 1.6e4 |
| engine | 2.0e-6 | 1.6e-4 | 0.020 | 0.24 | 0.62 | 0.93 | 0.995 | 0.9999 |
| mean cosine | 0 | 0 | 0.01 | 0.12 | 0.31 | 0.46 | 0.50 | 0.50 |

The leaves' absorption, the viscous and thermal boundary layers on their two faces,
matches a rigid wall's in closed form (Morse and Ingard 1968, 6.4) to 1e-12: 3.6e-4
of a leaf's area at 63 Hz, 1.5e-3 at 1 kHz, 4.1e-3 at 8 kHz, as the square root of
the frequency.

Then a wood of leaves 50 m deep over a ground that absorbs everything, the source and
listeners 1.5 m up, the listeners 5, 20 and 50 m behind it: the wood's insertion loss
(the level without it less the level with it, from two runs of the early paths and
100,000 late rays per band), compared with ISO 9613-2's dense-foliage attenuation for
50 m of foliage (Annex A, its rates per metre for 20 to 200 m), and with what the
early paths alone lose (the leaves' extinction):

| leaves | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | |
|---|---|---|---|---|---|---|---|---|---|
| ISO 9613-2, dense foliage | 1.00 | 1.50 | 2.00 | 2.50 | 3.00 | 4.00 | 4.50 | 6.00 | dB |
| 0.1 m²/m³, 5 cm wide | 0.01 | 0.01 | 0.02 | 0.02 | 0.06 | 0.45 | 4.66 | 16.49 | early paths' loss |
| | 0.01 | 0.01 | 0.02 | 0.02 | 0.05 | 0.31 | 2.92 | 8.76 | 5 m behind |
| | 0.01 | 0.01 | 0.02 | 0.02 | 0.05 | 0.36 | 3.48 | 10.47 | 50 m behind |
| 0.25 m²/m³, 5 cm wide | 0.02 | 0.03 | 0.04 | 0.06 | 0.14 | 1.11 | 11.65 | 41.22 | early paths' loss |
| | 0.02 | 0.03 | 0.04 | 0.06 | 0.12 | 0.79 | 7.06 | 17.33 | 5 m behind |
| | 0.02 | 0.03 | 0.04 | 0.06 | 0.13 | 0.90 | 8.17 | 19.98 | 50 m behind |
| 0.5 m²/m³, 5 cm wide | 0.04 | 0.06 | 0.08 | 0.12 | 0.29 | 2.23 | 23.30 | 82.45 | early paths' loss |
| | 0.04 | 0.06 | 0.08 | 0.12 | 0.24 | 1.57 | 11.88 | 32.19 | 5 m behind |
| | 0.04 | 0.06 | 0.08 | 0.12 | 0.26 | 1.78 | 14.79 | 30.69 | 50 m behind |
| 1 m²/m³, 5 cm wide | 0.08 | 0.11 | 0.16 | 0.24 | 0.57 | 4.45 | 46.61 | 164.90 | early paths' loss |
| | 0.08 | 0.11 | 0.16 | 0.23 | 0.49 | 3.08 | 20.12 | 36.15 | 5 m behind |
| | 0.08 | 0.11 | 0.16 | 0.24 | 0.51 | 3.37 | 23.22 | 39.98 | 50 m behind |
| 0.25 m²/m³, 2 cm wide | 0.02 | 0.03 | 0.04 | 0.06 | 0.08 | 0.14 | 0.58 | 5.92 | early paths' loss |
| | 0.02 | 0.03 | 0.04 | 0.06 | 0.08 | 0.13 | 0.45 | 4.21 | 5 m behind |
| | 0.02 | 0.03 | 0.04 | 0.06 | 0.08 | 0.13 | 0.50 | 4.78 | 50 m behind |
| 0.25 m²/m³, 10 cm wide | 0.02 | 0.03 | 0.04 | 0.12 | 1.08 | 11.61 | 41.16 | 53.29 | early paths' loss |
| | 0.02 | 0.03 | 0.04 | 0.10 | 0.75 | 6.78 | 15.88 | 19.27 | 5 m behind |
| | 0.02 | 0.03 | 0.04 | 0.11 | 0.85 | 7.99 | 18.41 | 22.07 | 50 m behind |

The standard's table is empirical, for the foliage of whole woods, so this compares
magnitudes rather than checking against an exact answer. Leaves 5 cm wide at 0.1 m²
per m³, about a wood's trunk space with some foliage in it, take about what the table
gives at 4 kHz (3 to 3.5 dB against 4.5) and more at 8 kHz (9 to 10 against 6); below
1 kHz they take next to nothing, where the table's 1 to 3 dB are the standard's
allowance for the whole wood, not what leaves scatter. The loss rises far more
steeply with frequency than the table, and with the leaves' area and width, as
measurements in dense foliage do (Aylor 1972 measured the loss through reeds and corn
rising by several decibels per octave above 500 Hz): a leaf scatters once its size
nears the wavelength, so wider
leaves take hold at lower frequencies, and leaves 2 cm wide hardly matter below
8 kHz. What the leaves scatter comes back as late sound, so a listener loses far less
than the early paths do (the extinction row): with 1 m² per m³ the early paths lose
165 dB at 8 kHz and the listener 36 to 40, the rest arriving scattered. The level
behind dense leaves comes with the late rays' noise, a few tenths of a decibel at
100,000 rays. No measurement of a known leaf area and width was available to check
against; the bridge between the disc's two limits, the discs standing in for real
leaves, which flex and hang in clumps, and the single phase function per band are
the model's assumptions.

## The wave solver in wind and temperature against a parabolic equation

`validation/wave_outdoor.py` checks the wave solver's speed of sound per cell against
the engine's parabolic equation (itself within 0.25 dB of the exact solution over grass
in still air, see [above](#ground-reflection-against-the-exact-solution)), marched in
the vertical plane through the source and the listener with the effective speed of
sound along it. Both take the same profiles; the wave solver steps the scene in three
dimensions. Over grass (Miki's impedance, 200 kPa s/m²), the source 2 m up, the
listeners 1.5 and 5 m up, 50 to 200 m away: the 63 Hz band from a grid for that band
(8 points per wavelength, 23 million cells), the 125 Hz band from one for it (32
million cells), up to 100 m. The ground reaches three times the half-width of the
reflection's first Fresnel zone at the farthest listener to either side, and 30 m past
it, so that its edges change little. Levels in dB over the free field, the mean square
of the pressure over the band's slices, wave solver / parabolic equation:

| 63 Hz | 50 m, 1.5 m up | 50 m, 5 m up | 100 m, 1.5 m up | 100 m, 5 m up | 200 m, 1.5 m up | 200 m, 5 m up |
|---|---|---|---|---|---|---|
| still | +5.2 / +5.2 | +4.3 / +4.4 | +5.0 / +5.2 | +4.2 / +4.4 | +5.0 / +5.0 | +4.0 / +4.0 |
| downwind | +5.2 / +5.8 | +3.9 / +4.4 | +5.8 / +6.3 | +4.2 / +5.0 | +7.0 / +7.2 | +5.0 / +5.4 |
| upwind | +4.4 / +4.6 | +4.0 / +4.3 | +3.5 / +3.8 | +3.4 / +3.4 | +1.7 / +1.8 | +1.6 / +1.6 |
| sunny day | +4.9 / +5.1 | +4.2 / +4.4 | +4.5 / +4.8 | +3.9 / +4.1 | +4.0 / +4.0 | +3.3 / +3.3 |
| clear night | +5.1 / +5.4 | +4.1 / +4.4 | +5.2 / +5.6 | +4.1 / +4.6 | +5.6 / +5.8 | +4.4 / +4.6 |

| 125 Hz | 50 m, 1.5 m up | 50 m, 5 m up | 100 m, 1.5 m up | 100 m, 5 m up |
|---|---|---|---|---|
| still | +2.6 / +2.8 | +0.6 / +0.8 | +1.9 / +1.9 | +0.2 / +0.3 |
| downwind | +2.9 / +3.3 | -0.5 / -0.2 | +2.6 / +2.9 | -1.1 / -0.9 |
| upwind | +2.4 / +2.1 | +1.6 / +1.4 | +0.2 / +0.1 | +0.4 / +0.4 |
| sunny day | +2.4 / +2.6 | +0.8 / +1.0 | +1.4 / +1.4 | +0.3 / +0.4 |
| clear night | +3.0 / +3.0 | +0.6 / +0.5 | +2.5 / +2.3 | +0.2 / +0.0 |

The weathers are the presets: downwind and upwind in the `windy` one (5 m/s 10 m up),
`sunny_day` and `clear_night`. The wave solver is within 0.7 dB of the parabolic
equation everywhere, 0.23 dB on average at 63 Hz and 0.17 dB at 125 Hz, and follows
the weather as it does: 2 dB louder downwind than in still air at 200 m, 3.3 dB
quieter upwind. At 63 Hz it comes out a little low downwind and under the night's
inversion (by up to 0.7 dB), where the wind and the temperature change fastest in the
lowest cells, and comes closer on a finer grid: downwind at 50 m, 63 Hz, 0.44 dB below at 8 points per
wavelength and 0.21 dB at 12 (in a narrower scene). Neither the deep shadow upwind (it
starts a few hundred metres away at these heights) nor reflections off buildings, where
the effective speed of sound along the horizontal from the source stands in for the
flow, were checked (moving air is, [below](#moving-air-in-the-wave-solver)).

## Moving air in the wave solver

`validation/wave_wind.py` checks the wave solver's moving air (`simulation.wave.wind:
moving`, see [How the simulation bends sound](outdoor.md#how-the-simulation-bends-sound))
against exact solutions, and the effective speed of sound beside it, in the 63 Hz band
(40 to 90 Hz, its edges a third of an octave wide) from a grid for that band.

**Free field in a uniform wind.** A wind of 15 m/s over a rigid ground, the source 2 m
up, listeners 1.5 m up 30 m away downwind, upwind, across the wind and at 45 degrees
to it. In a uniform wind each path (the direct sound and the ground's image) is the
convected Green's function, an impulse of height 1 / R* at (R* - M x) / (c (1 - M²)),
R* = √(x² + (1 - M²)(y² + z²)), x along the wind. The grid's own dispersion puts the
still air's sound 0.61 ms late along the grid's axes at 30 m (0.15 ms at 45 degrees),
its level within 0.4 dB of the exact one, and is the same with and without wind, so
each windy run is compared with the still air's run on the same grid, moved by the
exact change of its arrival and scaled by R / R*. Level and arrival over the arrival
(30 ms before to 50 ms after), and the waveform's error:

| 15 m/s, 30 m | downwind | upwind | across | 45 degrees |
|---|---|---|---|---|
| level, effective / moving (dB) | -0.90 / +0.02 | +0.79 / -0.04 | -0.28 / -0.00 | -0.75 / +0.00 |
| arrival, effective / moving (ms) | -0.03 / -0.07 | +0.03 / +0.09 | -0.15 / +0.00 | -0.04 / +0.00 |
| waveform's error, effective / moving (dB) | -19.8 / -30.1 | -20.1 / -28.0 | -22.9 / -65.1 | -21.4 / -45.0 |

Moving air is within 0.05 dB and 0.1 ms of the exact solution in every direction. The
effective speed of sound gets the arrival about right but not the level, up to 0.9 dB
off: a speed of sound that changes with the bearing is not a moving medium.

**A façade downwind.** A rigid building 40 m downwind of the source, its façade across
the wind, 30 m high and 50 m wide, and a listener 10 m upwind of the source, both 2 m
up: the façade sends the sound back against the wind. Ray theory in a uniform wind
gives its arrival (a ray goes at c n + v for its wave normal n; off the façade the wave
keeps its frequency and its wavenumber along the façade), 1.78 ms later than in still
air. Against the still air's run on the same grid:

| Façade echo | later than in still air | off by |
|---|---|---|
| exact (ray theory) | +1.78 ms | |
| effective speed of sound | -8.81 ms | -10.58 ms |
| moving air | +1.80 ms | +0.02 ms |

The effective speed of sound sends the echo back at the speed of sound going away from
the source, 80 v / (c² - v²) = 10.2 ms too soon over its 40 m back to the source;
moving air gets it within 0.02 ms. The wind is the uniform profile everywhere, the
building included: the check is of the sound in moving air, not of the flow round a
building, which the engine does not compute.

**A wind that grows with height.** The `windy` preset (5 m/s 10 m up, logarithmic) over
grass, downwind and upwind, the scene of [the check above](#the-wave-solver-in-wind-and-temperature-against-a-parabolic-equation),
against the parabolic equation with the effective speed of sound, which holds at
these grazing angles. Levels in dB over the free field, wave solver / parabolic
equation:

| 63 Hz | 50 m, 1.5 m up | 50 m, 5 m up | 100 m, 1.5 m up | 100 m, 5 m up | 200 m, 1.5 m up | 200 m, 5 m up |
|---|---|---|---|---|---|---|
| downwind, parabolic equation | +5.7 | +4.4 | +6.3 | +4.9 | +7.2 | +5.4 |
| downwind, effective | +5.2 | +3.9 | +5.8 | +4.3 | +7.0 | +5.0 |
| downwind, moving | +5.7 | +4.4 | +6.3 | +4.7 | +7.5 | +5.3 |
| upwind, parabolic equation | +4.6 | +4.3 | +3.8 | +3.4 | +1.8 | +1.6 |
| upwind, effective | +4.4 | +4.0 | +3.5 | +3.4 | +1.7 | +1.6 |
| upwind, moving | +4.5 | +4.2 | +3.5 | +3.4 | +1.9 | +1.5 |

Moving air is within 0.34 dB of the parabolic equation, 0.13 dB on average; the
effective speed of sound within 0.69 dB, 0.31 dB on average. Downwind, where the
effective speed of sound comes out about 0.5 dB low at 50 and 100 m, moving air is
within 0.05 dB on the same grid.

Moving air takes three to four times as long as the effective speed of sound: 23 s
against 7 s in the free field (3.2 million cells), 47 s against 15 s with the façade
(6 million) and 5 to 6 minutes against 1.5 with the preset (23 million), on 4 cores.

## Outdoor results against exact solutions

`validation/insertion_loss.py` runs whole outdoor scenes through the results an outdoor
run reports, the level relative to free field and the insertion loss (see
[Outdoor scenes](outdoor.md#results-outdoors)), and compares them with exact solutions.

A thin barrier 3 m high, 10 m from a source 1 m up, with listeners 1.5 m up, 2 to 90 m
behind it, over a ground that absorbs everything: its insertion loss, from the two
runs, against the Biot-Tolstoy-Medwin solution for a half-plane
([Edge diffraction](#edge-diffraction-against-the-exact-solution)), per octave band, and
Maekawa's chart for reference:

| listener | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | |
|---|---|---|---|---|---|---|---|---|---|
| 2 m behind | 8.84 | 11.09 | 13.67 | 16.49 | 19.44 | 22.43 | 25.43 | 28.44 | exact, dB |
| | -0.01 | -0.00 | +0.00 | +0.00 | +0.00 | +0.01 | +0.02 | +0.07 | engine - exact |
| | +0.22 | +0.06 | -0.05 | -0.15 | -0.23 | -0.29 | -0.33 | -0.35 | Maekawa - exact |
| 10 m behind | 8.32 | 10.06 | 12.16 | 14.63 | 17.38 | 20.30 | 23.28 | 26.28 | exact, dB |
| | -0.01 | -0.01 | -0.00 | +0.00 | +0.00 | +0.00 | +0.01 | +0.03 | engine - exact |
| | -1.14 | -1.35 | -1.43 | -1.47 | -1.54 | -1.62 | -1.68 | -1.71 | Maekawa - exact |
| 40 m behind | 8.16 | 9.69 | 11.58 | 13.87 | 16.50 | 19.35 | 22.31 | 25.30 | exact, dB |
| | -0.01 | -0.00 | -0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.02 | engine - exact |
| | -1.49 | -1.72 | -1.80 | -1.82 | -1.86 | -1.93 | -2.00 | -2.05 | Maekawa - exact |
| 90 m behind | 8.12 | 9.61 | 11.46 | 13.70 | 16.30 | 19.13 | 22.09 | 25.08 | exact, dB |
| | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | +0.00 | +0.01 | +0.02 | engine - exact |
| | -1.55 | -1.79 | -1.87 | -1.88 | -1.92 | -1.99 | -2.06 | -2.11 | Maekawa - exact |

The engine agrees within 0.07 dB; Maekawa's chart is within 2 dB.

The same barrier on a rigid ground, against the four paths over its top (the source
and its image in the ground, to the listener and its image), each by the exact
solution, and without the barrier the direct sound and its reflection, averaged over
each band. In the bands up to 500 Hz the engine reads the early paths through band
filters, whose skirts let in a little of the neighbouring bands where four paths
interfere, so the table also compares it with the exact solution read through the same
filters (*through the filters*; from 1 kHz up the engine averages each pair of paths'
interference over the band exactly, and the row repeats the one above it). First with
`simulation.early.diffraction_reflections: both_sides`, which takes all four paths:

| listener | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | |
|---|---|---|---|---|---|---|---|---|---|
| 2 m behind | 14.21 | 11.15 | 23.60 | 10.22 | 19.58 | 22.62 | 24.84 | 28.15 | exact, dB |
| | -0.57 | +0.23 | -2.11 | +0.38 | +0.01 | +0.00 | +0.04 | +0.15 | engine - exact |
| | -0.02 | -0.06 | -0.06 | -0.02 | +0.01 | +0.00 | +0.04 | +0.15 | through the filters |
| | +0.00 | +0.00 | +0.00 | +0.01 | -0.01 | -0.00 | -0.01 | -0.01 | without the barrier |
| 10 m behind | 6.31 | 13.90 | 25.19 | 14.78 | 9.06 | 22.29 | 22.66 | 25.94 | exact, dB |
| | +0.05 | -0.05 | -2.22 | -0.05 | +0.01 | +0.00 | +0.04 | +0.07 | engine - exact |
| | -0.04 | -0.08 | -0.03 | -0.04 | +0.01 | +0.00 | +0.04 | +0.07 | through the filters |
| | +0.00 | +0.00 | +0.00 | +0.01 | +0.00 | -0.00 | +0.00 | -0.02 | without the barrier |
| 40 m behind | 4.41 | 8.42 | 18.06 | 18.55 | 18.19 | 15.47 | 21.19 | 24.47 | exact, dB |
| | +0.03 | +0.08 | -0.51 | +0.18 | +0.00 | +0.01 | +0.00 | +0.05 | engine - exact |
| | -0.04 | -0.07 | -0.07 | -0.06 | +0.00 | +0.01 | +0.00 | +0.05 | through the filters |
| | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.01 | -0.01 | -0.01 | without the barrier |
| 90 m behind | 4.17 | 7.93 | 16.66 | 12.67 | 20.67 | 24.87 | 16.25 | 23.53 | exact, dB |
| | +0.03 | +0.08 | -0.60 | +0.25 | +0.01 | +0.01 | +0.02 | +0.06 | engine - exact |
| | -0.04 | -0.07 | -0.06 | -0.04 | +0.01 | +0.01 | +0.02 | +0.06 | through the filters |
| | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.01 | -0.01 | without the barrier |

Read alike, the engine is within 0.15 dB of the exact solution at every listener and
band (0.08 dB up to 500 Hz, 0.15 dB at 8 kHz, 0.04 dB elsewhere), and without the
barrier (the direct sound and its reflection, added in pressure) within 0.02 dB. The
2 dB at 250 Hz near the barrier is what the band filters let in from the bands beside
a deep interference dip, as a measured response read through the same filters would
show, not a gap in the model.

Then with the default `one_side`, where a path reflects before or after the edge but
not both, so the path reflected both before and after the barrier, a quarter of the
sound over it, is left out:

| listener | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | |
|---|---|---|---|---|---|---|---|---|---|
| 2 m behind | -2.31 | +1.46 | -3.87 | +0.89 | +0.34 | +0.30 | +0.51 | +0.54 | engine - exact |
| | -1.77 | +1.17 | -1.81 | +0.49 | +0.34 | +0.30 | +0.51 | +0.54 | through the filters |
| 10 m behind | +0.92 | -1.66 | -5.03 | +0.88 | +0.53 | +0.43 | +0.65 | +0.54 | engine - exact |
| | +0.83 | -1.69 | -2.84 | +0.89 | +0.53 | +0.43 | +0.65 | +0.54 | through the filters |
| 40 m behind | +1.70 | +0.71 | -2.69 | +1.78 | +0.41 | +0.27 | +0.71 | +0.61 | engine - exact |
| | +1.64 | +0.56 | -2.26 | +1.53 | +0.41 | +0.27 | +0.71 | +0.61 | through the filters |
| 90 m behind | +1.85 | +1.05 | -2.42 | +1.74 | -0.52 | -0.74 | +0.77 | +0.67 | engine - exact |
| | +1.78 | +0.91 | -1.88 | +1.46 | -0.52 | -0.74 | +0.77 | +0.67 | through the filters |

Without that path the insertion loss comes out 0.3 to 0.8 dB too high from 1 kHz up,
and in the bands below, where the four paths interfere, up to 3 dB off either way.
Over grass and other soft grounds, whose reflections near grazing are weak, the gap is
smaller. The sound that goes over the top, down to the ground and over the top again
is left out of both.

A source 1 m over grass and nothing else, the listener 1.5 m up: the level relative to
free field against the exact solution for a locally reacting ground
([Ground reflection](#ground-reflection-against-the-exact-solution)), averaged over each
band:

| distance | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | |
|---|---|---|---|---|---|---|---|---|---|
| 10 m | 5.27 | 3.74 | -1.23 | -1.55 | 1.43 | 0.27 | 1.10 | 0.85 | exact, dB |
| | -0.04 | -0.11 | -0.06 | +0.44 | -0.01 | -0.00 | -0.01 | -0.01 | engine - exact |
| 50 m | 5.52 | 3.58 | -3.02 | -11.16 | -2.05 | 3.84 | 2.74 | 2.43 | exact, dB |
| | -0.07 | -0.15 | -0.03 | +0.85 | +0.00 | -0.00 | -0.00 | -0.00 | engine - exact |
| 200 m | 5.31 | 1.10 | -13.10 | -22.37 | -12.89 | -5.56 | 0.52 | 4.91 | exact, dB |
| | -0.15 | -0.10 | +0.87 | +0.72 | +0.00 | +0.00 | -0.00 | -0.00 | engine - exact |

From 1 kHz up the engine averages each pair of paths' interference over the band
exactly and agrees within 0.01 dB; in the bands up to 500 Hz it reads the early paths
through band filters, whose skirts fill the ground dip by up to about 0.9 dB, as in the
ground's own check above. Both results are checked against theory, not against
measurements of a barrier or a ground.

## Outdoor sound in wind against a fast field program

`validation/ffp.py` is a fast field program for sound over a flat ground through a
layered atmosphere, the kind of model the benchmark cases for outdoor sound of
Attenborough and others (J. Acoust. Soc. Am. 97, 1995, 173-191) were solved with,
written apart from the engine to check its parabolic equation in wind and temperature.
It writes the sound field as an integral over the horizontal wavenumber and, for each
wavenumber, solves the vertical problem exactly through a stack of layers in each of
which the speed of sound is constant: the atmosphere as a staircase of layers a quarter
wavelength thick, 1 m at most, halving sixteen times towards the ground, the source,
the listener and the top; the ground an impedance boundary; and above the top, 1000 m
up, where the engine's atmosphere stops changing, homogeneous air that sends nothing
back. The integral follows a contour dipped into the complex plane past the poles near
the real axis, with the Bessel function evaluated exactly rather than by its far-field
form, so the field is right at 100 m as at 10 km. Its Bessel functions are its own (a
power series up to an argument of 17 and the Hankel expansion beyond, 2 × 10⁻¹⁰ apart
where they meet and 2 × 10⁻⁸ off tabulated integrals), so the script needs nothing but
numpy.

In still air it is checked against the exact solution for a source over a locally
reacting ground ([Ground reflection](#ground-reflection-against-the-exact-solution)),
the source 5 m and the listener 1 m up, the level over the free field's:

| | 100 m | 300 m | 1 km | 3 km | 10 km | |
|---|---|---|---|---|---|---|
| 10 Hz, rigid | 6.02 | 6.02 | 6.02 | 6.02 | 6.02 | exact, dB |
| | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | ffp - exact |
| 10 Hz, grass | 6.04 | 6.12 | 6.26 | 6.43 | 6.51 | exact, dB |
| | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | ffp - exact |
| 100 Hz, rigid | 5.98 | 6.02 | 6.02 | 6.02 | 6.02 | exact, dB |
| | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | ffp - exact |
| 100 Hz, grass | 3.12 | 0.72 | -7.95 | -23.91 | -35.29 | exact, dB |
| | 0.000 | 0.000 | 0.000 | 0.000 | -0.001 | ffp - exact |
| 1000 Hz, rigid | 1.72 | 5.61 | 5.98 | 6.02 | 6.02 | exact, dB |
| | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | ffp - exact |
| 1000 Hz, grass | 1.30 | -6.58 | -16.61 | -26.04 | -36.46 | exact, dB |
| | 0.000 | 0.000 | 0.000 | 0.000 | -0.001 | ffp - exact |

It agrees to 0.001 dB from 10 Hz to 1 kHz and from 100 m to 10 km, over a rigid ground
and over grass. With the speed of sound rising 0.1 m/s per metre of height, halving its
layers changes the level by 0.006 dB at most (100 Hz and 1 kHz, 1 and 10 km out), so the
staircase is converged. In a deep shadow its layers scatter a little sound: the field
has a floor near -60 dB re free field, lower with thinner layers, so nothing is compared
below -60 dB, where nothing is heard anyway.

`validation/benchmark_1995.py` runs the benchmark's cases 1 to 3 through the program
and the engine: a source 5 m and a listener 1 m over a ground of flow resistivity
200 kPa s/m² (Delany and Bazley's impedance), 100 m to 10 km apart, in still air
(case 1), with the speed of sound rising 0.1 m/s per metre of height (case 2, downward
refraction: downwind or under an inversion) and falling as much (case 3, upward
refraction: upwind or over sun-warmed ground). The benchmark's gradients go on without
end; the engine's atmosphere stops changing 1000 m up, 100 m/s off the ground's speed,
and the program follows it there. Such a gradient over a kilometre of height is beyond
any weather, so two more cases stop it 100 m up, 10 m/s more or less than at the
ground, as a strong wind blows: the weather the engine is meant for. Case 4, case 3
with turbulence, is not run: the benchmark's eddies have a Gaussian spectrum, the
engine's a von Kármán one, checked in
[Turbulence](#turbulence-against-a-parabolic-equation-through-random-eddies).

First the engine's parabolic equations alone, at the benchmark's pure tones: the
narrow-angle one (`acoustic_spaces.engine.parabolic`, the Green's function parabolic
equation of Gilbert and Di, the default, which gives the engine's sound over the open
ground in the bands up to 500 Hz and in a shadow up to 2 kHz) and the wide-angle one
(`acoustic_spaces.engine.wide_angle`, see
[The wide-angle parabolic equation](#the-wide-angle-parabolic-equation)); the program's
level re free field and each equation's difference from it (`-`: deeper than -60 dB):

| | 100 m | 200 m | 500 m | 1 km | 2 km | 5 km | 10 km | |
|---|---|---|---|---|---|---|---|---|
| case 1, still air, 10 Hz | 6.00 | 6.02 | 6.04 | 6.06 | 6.08 | 6.12 | 6.12 | ffp, dB |
| | +0.04 | +0.01 | -0.03 | -0.03 | -0.04 | -0.04 | -0.01 | narrow - ffp |
| | -0.02 | -0.00 | +0.00 | +0.00 | +0.00 | -0.00 | +0.03 | wide - ffp |
| case 1, still air, 100 Hz | 4.26 | 3.72 | 2.27 | -0.11 | -4.88 | -16.72 | -24.77 | ffp, dB |
| | -0.03 | -0.02 | -0.00 | +0.02 | +0.02 | -0.05 | -0.10 | narrow - ffp |
| | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 | -0.04 | wide - ffp |
| case 1, still air, 1000 Hz | 0.72 | -3.88 | -11.14 | -16.96 | -22.88 | -30.78 | -36.78 | ffp, dB |
| | -0.03 | -0.04 | -0.05 | -0.05 | -0.05 | -0.05 | -0.05 | narrow - ffp |
| | -0.00 | -0.00 | -0.00 | +0.00 | -0.00 | +0.00 | +0.01 | wide - ffp |
| case 2, downward refraction 0.1/s, 10 Hz | 6.23 | 6.67 | 8.62 | 13.02 | 13.84 | 20.29 | 24.28 | ffp, dB |
| | -0.20 | -0.27 | -0.38 | -0.45 | +0.14 | -6.47 | -2.10 | narrow - ffp |
| | -0.02 | -0.00 | +0.00 | +0.00 | -0.01 | -0.01 | +0.03 | wide - ffp |
| case 2, downward refraction 0.1/s, 100 Hz | 4.60 | 4.84 | 5.51 | 2.81 | 8.72 | 3.92 | 11.89 | ffp, dB |
| | -0.04 | -0.04 | -0.01 | -0.03 | -0.02 | +5.52 | -19.77 | narrow - ffp |
| | +0.00 | +0.00 | -0.00 | -0.00 | -0.01 | -0.02 | -0.01 | wide - ffp |
| case 2, downward refraction 0.1/s, 1000 Hz | 2.13 | 1.71 | 8.30 | 9.25 | 5.31 | -2.51 | -10.13 | ffp, dB |
| | -0.04 | -0.04 | -0.04 | +0.32 | -10.92 | +3.40 | -0.58 | narrow - ffp |
| | -0.00 | -0.00 | -0.00 | -0.00 | -0.01 | -0.05 | -0.11 | wide - ffp |
| case 3, upward refraction 0.1/s, 10 Hz | 5.77 | 5.35 | 3.40 | -1.29 | -13.32 | -41.23 | -77.99 | ffp, dB |
| | +0.28 | +0.21 | +0.21 | +0.47 | -0.06 | -9.08 | - | narrow - ffp |
| | -0.02 | -0.00 | -0.00 | -0.01 | -0.01 | +0.28 | - | wide - ffp |
| case 3, upward refraction 0.1/s, 100 Hz | 3.88 | 2.40 | -3.54 | -16.07 | -44.04 | -64.00 | -98.16 | ffp, dB |
| | -0.02 | -0.01 | +0.01 | +0.04 | -0.12 | - | - | narrow - ffp |
| | -0.00 | -0.00 | -0.01 | -0.01 | -0.05 | - | - | wide - ffp |
| case 3, upward refraction 0.1/s, 1000 Hz | -1.08 | -12.11 | -52.50 | -92.86 | -98.39 | -94.54 | -121.03 | ffp, dB |
| | -0.05 | -0.04 | +7.79 | - | - | - | - | narrow - ffp |
| | -0.00 | -0.00 | -0.02 | - | - | - | - | wide - ffp |
| 0.1/s up to 100 m, 10 Hz | 6.23 | 6.68 | 8.55 | 13.37 | 15.64 | 18.42 | 18.27 | ffp, dB |
| | -0.20 | -0.24 | -0.33 | -0.49 | -0.34 | -0.38 | -0.49 | narrow - ffp |
| | -0.02 | -0.00 | +0.00 | +0.00 | -0.00 | -0.00 | -0.01 | wide - ffp |
| 0.1/s up to 100 m, 100 Hz | 4.60 | 4.84 | 5.51 | 2.80 | -13.83 | -2.11 | -10.57 | ffp, dB |
| | -0.04 | -0.04 | -0.02 | +0.04 | +3.22 | -0.19 | -3.12 | narrow - ffp |
| | +0.00 | +0.00 | -0.00 | -0.00 | +0.02 | -0.00 | +0.00 | wide - ffp |
| 0.1/s up to 100 m, 1000 Hz | 2.13 | 1.71 | 8.30 | 9.25 | 0.48 | -21.83 | -47.12 | ffp, dB |
| | -0.04 | -0.04 | -0.04 | +0.33 | +1.23 | -1.93 | +11.32 | narrow - ffp |
| | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | +0.02 | wide - ffp |
| -0.1/s up to 100 m, 10 Hz | 5.78 | 5.35 | 3.28 | -0.47 | -9.92 | -44.43 | -69.88 | ffp, dB |
| | +0.28 | +0.24 | +0.32 | +0.28 | +0.22 | +1.42 | - | narrow - ffp |
| | -0.02 | -0.00 | -0.00 | -0.01 | -0.01 | -0.41 | - | wide - ffp |
| -0.1/s up to 100 m, 100 Hz | 3.88 | 2.40 | -3.53 | -16.14 | -45.75 | -102.87 | -98.15 | ffp, dB |
| | -0.02 | -0.01 | -0.02 | +0.10 | -5.79 | - | - | narrow - ffp |
| | -0.00 | -0.00 | -0.01 | -0.02 | +0.07 | - | - | wide - ffp |
| -0.1/s up to 100 m, 1000 Hz | -1.08 | -12.11 | -52.51 | -64.37 | -98.31 | -125.03 | -121.03 | ffp, dB |
| | -0.04 | -0.05 | +0.33 | - | - | - | - | narrow - ffp |
| | -0.00 | -0.00 | -0.01 | - | - | - | - | wide - ffp |

In still air the narrow-angle equation and the program agree within 0.05 dB to 5 km and
0.1 dB at 10 km; at 10 Hz, below the engine's lowest band, the narrow-angle equation is
0.2 to 0.5 dB off in every case. In
wind the agreement holds as far as the sound reaches the listener at the flat angles a
parabolic equation carries. With the gradient stopping 100 m up, which turns back only
the sound within 14 degrees of the horizontal, the parabolic equation is within 0.05 dB
at 100 Hz and 0.35 dB at 1 kHz up to 1 km downwind, 1 to 3 dB at 2 km, where the bent
paths interfere and the two place a dip a little apart, within 2 dB at 5 km, and 3 dB
at 10 km at 100 Hz (11 dB at 1 kHz, 47 dB down, a band the engine's rays give
downwind anyway). Upwind, in the shadow, it is within 0.1 dB until the sound is 20 dB
down and strays by 6 dB (100 Hz, 2 km, 46 dB down) to 8 dB (1 kHz, 500 m, 52 dB down)
deeper in, below where the eddies keep a shadow in practice (see
[Turbulence](outdoor.md#turbulence)). With the benchmark's gradient going on to 1000 m,
sound that has risen hundreds of metres comes back down at 17 degrees and more, and
upwind the speed of sound a kilometre up is 100 m/s below the ground's, an index of
refraction of 1.4; the parabolic equation, a narrow-angle one that marches the field
out along the ground and takes the refraction as a phase shift at each step, carries
neither: downwind the two part beyond 1 km at 1 kHz and 2 km at 100 Hz, by 3 to 20 dB
at a pure tone where several bent paths interfere, and upwind it comes out 8 dB too
loud at 1 kHz 500 m into the shadow (0.3 dB with the gradient stopping at 100 m). Such
profiles are beyond any weather; the engine's atmosphere is meant for the gradients
near the ground that make the shadows and the focusing in practice, and within a few
kilometres of the source. The wide-angle equation carries both: within 0.05 dB of the
program in every case, at every tone and to 10 km, wherever the sound is within 40 dB
of free field, but for 0.11 dB at 1 kHz 10 km downwind; deeper in the shadow within
0.07 dB at 100 Hz and 1 kHz, and 0.3 to 0.4 dB at 10 Hz 5 km upwind, 41 to 44 dB down.

Then whole scenes, each case as an outdoor scene: the gradient as a measured
atmosphere (`atmosphere.table`) with the wind along the line of the listeners, the air
at 19.6 °C (343 m/s at the ground) and 50 % humidity, and the engine's defaults
otherwise (the early paths as image sources up to one reflection, 200 late rays). The
level relative to free field the engine reports per octave band from 63 Hz to 1 kHz
([Outdoor scenes](outdoor.md#results-outdoors)) against the program averaged over the
band as the engine reads it (eight frequencies per band; 24 at 1 kHz, where the rays
average the paths' interference exactly), and in the bands up to 500 Hz also against the
program read through the engine's band filters, as the barrier check above is (*through
the filters*). The engine's rays carry the air's absorption along their own paths and the
program runs without it, a difference of the path lengths only, under 0.01 dB here:

**case 1, still air**

| distance | 63 | 125 | 250 | 500 | 1k | |
|---|---|---|---|---|---|---|
| 100 m | 5.17 | 3.15 | -2.31 | -6.43 | 1.37 | ffp, dB |
| | -0.05 | -0.11 | -0.03 | +0.59 | -0.00 | engine - ffp |
| | +0.00 | -0.01 | -0.01 | -0.00 | -0.00 | through the filters |
| 200 m | 5.01 | 2.07 | -6.18 | -11.08 | -2.96 | ffp, dB |
| | -0.08 | -0.14 | +0.17 | +0.54 | -0.00 | engine - ffp |
| | -0.02 | -0.01 | +0.03 | -0.03 | -0.00 | through the filters |
| 500 m | 4.58 | -0.53 | -14.60 | -18.32 | -10.16 | ffp, dB |
| | -0.14 | -0.04 | +0.93 | +0.44 | -0.00 | engine - ffp |
| | -0.04 | +0.05 | +0.16 | -0.11 | -0.00 | through the filters |
| 1 km | 3.85 | -3.98 | -22.32 | -24.12 | -15.97 | ffp, dB |
| | -0.15 | +0.20 | +1.47 | +0.31 | -0.00 | engine - ffp |
| | -0.02 | +0.12 | +0.32 | -0.23 | -0.00 | through the filters |
| 2 km | 2.41 | -9.47 | -28.84 | -30.03 | -21.90 | ffp, dB |
| | -0.10 | +0.81 | +1.30 | +0.11 | -0.00 | engine - ffp |
| | +0.04 | +0.27 | +0.53 | -0.42 | -0.00 | through the filters |

**case 2, downward refraction 0.1/s**

| distance | 63 | 125 | 250 | 500 | 1k | |
|---|---|---|---|---|---|---|
| 100 m | 5.55 | 3.42 | -2.27 | -5.22 | 2.55 | ffp, dB |
| | -0.05 | -0.05 | -0.04 | -0.07 | +0.04 | engine - ffp |
| | +0.00 | +0.06 | -0.04 | -0.68 | +0.04 | through the filters |
| 200 m | 6.21 | 2.97 | -5.67 | -6.00 | 2.28 | ffp, dB |
| | -0.04 | -0.04 | -0.05 | -0.07 | -0.23 | engine - ffp |
| | +0.03 | +0.09 | -0.27 | -0.69 | -0.23 | through the filters |
| 500 m | 9.03 | 1.29 | -2.98 | -2.74 | 5.07 | ffp, dB |
| | -0.03 | -0.04 | -0.06 | -0.05 | -1.78 | engine - ffp |
| | +0.12 | -0.19 | -0.12 | -0.82 | -1.78 | through the filters |
| 1 km | 6.79 | 1.37 | 1.43 | 1.55 | 6.63 | ffp, dB |
| | +0.03 | -0.05 | -0.04 | -0.11 | +0.48 | engine - ffp |
| | -0.22 | -0.15 | +0.12 | -0.19 | +0.48 | through the filters |
| 2 km | 5.93 | 5.97 | -0.51 | 4.14 | 4.18 | ffp, dB |
| | -0.02 | +0.07 | -0.53 | -0.75 | -0.77 | engine - ffp |
| | -0.32 | +0.34 | -1.32 | -0.45 | -0.77 | through the filters |

**case 3, upward refraction 0.1/s**

| distance | 63 | 125 | 250 | 500 | 1k | |
|---|---|---|---|---|---|---|
| 100 m | 4.78 | 2.82 | -2.44 | -7.68 | -0.30 | ffp, dB |
| | -0.01 | -0.02 | -0.09 | -0.06 | -0.17 | engine - ffp |
| | +0.04 | +0.08 | -0.04 | -0.60 | -0.17 | through the filters |
| 200 m | 3.72 | 0.85 | -7.14 | -16.62 | -11.46 | ffp, dB |
| | -0.01 | -0.02 | -0.02 | -0.07 | -0.05 | engine - ffp |
| | +0.05 | +0.11 | -0.11 | -0.53 | -0.05 | through the filters |
| 500 m | -0.84 | -6.20 | -20.67 | -44.90 | -52.70 | ffp, dB |
| | -0.01 | -0.00 | -0.02 | -0.10 | -0.07 | engine - ffp |
| | +0.08 | +0.08 | -0.88 | -2.33 | -0.07 | through the filters |
| 1 km | -10.83 | -19.87 | -43.44 | -90.90 | -100.23 | ffp, dB |
| | -0.01 | +0.01 | +0.02 | - | - | engine - ffp |
| | +0.06 | -0.12 | -3.41 | - | - | through the filters |
| 2 km | -33.35 | -48.82 | -78.33 | -88.90 | -94.67 | ffp, dB |
| | +0.01 | +0.02 | - | - | - | engine - ffp |
| | -0.11 | -0.91 | - | - | - | through the filters |

**0.1/s up to 100 m**

| distance | 63 | 125 | 250 | 500 | 1k | |
|---|---|---|---|---|---|---|
| 100 m | 5.55 | 3.42 | -2.27 | -5.22 | 2.55 | ffp, dB |
| | -0.05 | -0.05 | -0.04 | -0.06 | +0.04 | engine - ffp |
| | +0.00 | +0.06 | -0.04 | -0.68 | +0.04 | through the filters |
| 200 m | 6.21 | 2.97 | -5.67 | -6.00 | 2.28 | ffp, dB |
| | -0.04 | -0.04 | -0.05 | -0.07 | -0.23 | engine - ffp |
| | +0.03 | +0.09 | -0.27 | -0.69 | -0.23 | through the filters |
| 500 m | 9.03 | 1.29 | -2.98 | -2.74 | 5.07 | ffp, dB |
| | -0.03 | -0.03 | -0.06 | -0.05 | -1.78 | engine - ffp |
| | +0.12 | -0.18 | -0.12 | -0.82 | -1.78 | through the filters |
| 1 km | 6.74 | 1.38 | 1.43 | 1.55 | 6.63 | ffp, dB |
| | +0.04 | -0.04 | -0.04 | -0.11 | +0.48 | engine - ffp |
| | -0.21 | -0.14 | +0.13 | -0.19 | +0.48 | through the filters |
| 2 km | 3.80 | -1.16 | -2.85 | -5.93 | 1.15 | ffp, dB |
| | -0.07 | +0.01 | -0.09 | +0.44 | -0.97 | engine - ffp |
| | -0.42 | -0.16 | -0.00 | -0.46 | -0.97 | through the filters |

**-0.1/s up to 100 m**

| distance | 63 | 125 | 250 | 500 | 1k | |
|---|---|---|---|---|---|---|
| 100 m | 4.78 | 2.82 | -2.44 | -7.68 | -0.30 | ffp, dB |
| | -0.02 | -0.02 | -0.03 | -0.07 | -0.17 | engine - ffp |
| | +0.03 | +0.08 | +0.01 | -0.60 | -0.17 | through the filters |
| 200 m | 3.72 | 0.85 | -7.14 | -16.62 | -11.46 | ffp, dB |
| | -0.01 | -0.02 | -0.03 | -0.08 | -0.05 | engine - ffp |
| | +0.05 | +0.11 | -0.12 | -0.53 | -0.05 | through the filters |
| 500 m | -0.84 | -6.20 | -20.66 | -44.81 | -52.68 | ffp, dB |
| | -0.01 | -0.00 | -0.04 | -0.21 | -0.11 | engine - ffp |
| | +0.08 | +0.08 | -0.89 | -2.40 | -0.11 | through the filters |
| 1 km | -10.87 | -19.88 | -43.62 | -66.43 | -66.91 | ffp, dB |
| | -0.02 | -0.09 | +1.46 | - | - | engine - ffp |
| | +0.03 | -0.23 | -1.96 | - | - | through the filters |
| 2 km | -31.61 | -43.55 | -69.69 | -91.23 | -98.25 | ffp, dB |
| | +0.01 | +0.11 | - | - | - | engine - ffp |
| | +0.14 | +0.23 | - | - | - | through the filters |

In still air the early paths are image sources read through the band filters, and the
row *through the filters* is the one to compare: the engine is within 0.04 dB of the program up
to 200 m and within 0.3 dB at 1 km, 0.5 dB at 2 km, where the ground dip at 250 Hz is
29 dB deep and the engine's reflection factor is an approximation (see
[Ground reflection](#ground-reflection-against-the-exact-solution)); from 1 kHz up,
where the rays average the paths' interference over the band exactly, within
0.005 dB. In wind the bands up to 500 Hz come from the parabolic equation, whose band
levels are the averages over its eight frequencies, and the engine reports them as
computed, so there the plain average is the row to read: within 0.11 dB to 1 km in
every case, downwind as upwind, and at 2 km downwind 0.5 to 0.8 dB with the gradient
going on to 1000 m, where the pure tones above show the parabolic equation itself
parting from the program, and 0.1 (0.4 at 500 Hz) with it stopping at 100 m. Upwind
the engine follows the program into the shadow within 0.1 dB to 500 m and 45 dB below
free field, and within 1.5 dB at 1 km with the gradient stopping at 100 m (250 Hz,
44 dB down, where the pure tones above show the parabolic equation straying deep in
the shadow); the comparison stops 60 dB down. The 1 kHz band comes from the rays: within
0.5 dB but for 1.8 dB at 500 m downwind, under a caustic (the rays' own check allows
1.5 dB there, see [Outdoor scenes](outdoor.md#how-the-simulation-bends-sound)), and
1 dB at 2 km, where the bent paths crowd.

This comparison found two faults at the join between the parabolic equation's bands
and the rays', fixed in the same change (see the changelog): the response had a notch
up to 3 dB deep around 700 Hz (around 2.8 kHz in the penumbra of a shadow), as the
join blended spectra that stand for different sound, the whole field over the ground
on one path and each ray's own on the others, and the 500 Hz band of the level
relative to free field came out 0.5 to 1.2 dB too low, as the silent paths' higher
bands counted against its reading. Each band of the response now keeps its own
spectrum into the crossovers, and a path that carries nothing in a band is left out
of that band's reading; rooms change by under a hundred-thousandth of the response's
peak, and the still-air tables above by up to 0.2 dB in one cell.

## The wide-angle parabolic equation

With `simulation.early.parabolic_equation: wide_angle` (`--parabolic-equation
wide_angle`, **Parabolic equation** in the editor) the bands the parabolic equation
gives outdoors in wind or temperature gradients (see
[Wind and temperature](outdoor.md#wind-and-temperature)) come from
`acoustic_spaces.engine.wide_angle` instead of the narrow-angle default: the
split-step Padé parabolic equation of Collins (J. Acoust. Soc. Am. 93, 1993, 1736, as
in his program RAM). It marches the field with the one-way operator of the Helmholtz
equation, exact at every angle in air that does not change along the way, written as a
rational function of ten terms that never amplifies the waves that do not propagate
(RAM's stability constraint), in steps of three wavelengths; the heights are finite
differences of fourth order, twenty points per wavelength, with the ground's impedance
as the lower boundary, also to fourth order, and the field starts as the operator has
it (Collins' self-starter, J. Acoust. Soc. Am. 92, 1992, 2069). The ten tridiagonal
solves of each step run in native code, the frequencies of a band on the run's
threads.

`tests/test_wide_angle.py` checks each part on every change: one step against the
exact exponential of the operator, within 10⁻⁷ for sound up to 45 degrees from the
horizontal and 10⁻¹³ within 18 degrees of it; that it damps the waves that do not
propagate; the starting field's rational function within 10⁻⁸ of its own; the native
solves against dense ones, and on any number of threads bit for bit; in still air the
complex pressure over grass within 0.5 % (0.04 dB, 0.3 degrees) of the exact solution
at 125 and 500 Hz; the field 45 degrees up within 0.15 dB of the exact one, where the
narrow-angle equation's starting field falls 1 dB short; and the benchmark's case 2 at
100 Hz within 0.05 dB of the fast field program to 5 km, where the narrow-angle
equation is 6 dB off.

**The benchmark.** At the 1995 benchmark's pure tones (the table in
[Outdoor sound in wind](#outdoor-sound-in-wind-against-a-fast-field-program)) the
wide-angle equation is within 0.05 dB of the fast field program in every case, from
10 Hz to 1 kHz and from 100 m to 10 km, wherever the sound is within 40 dB of free
field, but for 0.11 dB at 1 kHz 10 km downwind; deeper in the shadow it is within
0.07 dB at 100 Hz and 1 kHz, and 0.3 to 0.4 dB at 10 Hz 5 km upwind, 41 to 44 dB down.
With the benchmark's gradients going on to a kilometre up, where the narrow-angle
equation strays by 3 to 20 dB beyond 1 to 2 km downwind and by 8 dB 500 m into the
shadow upwind at 1 kHz, the wide-angle one stays within 0.11 dB.

**The weather presets.** The gradients the engine is meant for change fastest near the
ground. Both equations against the fast field program in the presets: a source 2 m and
a listener 1.5 m up over the benchmark's ground, at 20 °C, 50 to 300 m apart, the
level re free field (dB) and each equation's difference from it at the octave bands'
centre tones:

| | 50 m | 100 m | 200 m | 300 m | |
|---|---|---|---|---|---|
| `windy`, downwind, 63 Hz | 6.14 | 6.91 | 8.24 | 9.32 | ffp, dB |
| | -0.19 | -0.19 | -0.16 | -0.18 | narrow - ffp |
| | +0.02 | +0.04 | +0.05 | +0.06 | wide - ffp |
| `windy`, downwind, 125 Hz | 4.76 | 5.33 | 5.66 | 4.67 | ffp, dB |
| | +0.10 | +0.10 | -0.02 | -0.11 | narrow - ffp |
| | +0.00 | +0.00 | -0.02 | -0.07 | wide - ffp |
| `windy`, downwind, 250 Hz | -2.27 | -6.77 | -26.18 | -22.74 | ffp, dB |
| | -0.16 | -0.27 | -0.04 | -0.42 | narrow - ffp |
| | -0.07 | -0.15 | -0.05 | -0.18 | wide - ffp |
| `windy`, downwind, 500 Hz | -6.43 | -6.84 | -3.53 | 0.20 | ffp, dB |
| | -0.07 | -0.05 | +0.06 | +0.06 | narrow - ffp |
| | +0.03 | +0.03 | +0.04 | +0.04 | wide - ffp |
| `windy`, upwind, 63 Hz | 4.86 | 3.93 | 1.91 | -0.24 | ffp, dB |
| | +0.19 | +0.10 | -0.07 | -0.17 | narrow - ffp |
| | -0.03 | -0.05 | -0.09 | -0.12 | wide - ffp |
| `windy`, upwind, 125 Hz | 3.41 | 1.58 | -2.42 | -6.73 | ffp, dB |
| | -0.23 | -0.31 | -0.37 | -0.48 | narrow - ffp |
| | -0.01 | -0.03 | -0.07 | -0.12 | wide - ffp |
| `windy`, upwind, 250 Hz | -0.29 | -4.13 | -12.53 | -21.73 | ffp, dB |
| | -0.05 | -0.10 | -0.20 | -0.30 | narrow - ffp |
| | +0.02 | +0.01 | -0.01 | -0.05 | wide - ffp |
| `windy`, upwind, 500 Hz | -6.16 | -14.45 | -33.95 | -56.97 | ffp, dB |
| | -0.16 | -0.16 | -0.14 | -0.12 | narrow - ffp |
| | +0.05 | +0.06 | +0.05 | +0.02 | wide - ffp |
| `sunny_day`, 63 Hz | 5.32 | 5.02 | 4.36 | 3.66 | ffp, dB |
| | +0.07 | +0.05 | +0.02 | +0.00 | narrow - ffp |
| | -0.01 | -0.01 | -0.02 | -0.03 | wide - ffp |
| `sunny_day`, 125 Hz | 3.96 | 3.18 | 1.57 | -0.03 | ffp, dB |
| | -0.08 | -0.09 | -0.07 | -0.08 | narrow - ffp |
| | -0.00 | -0.01 | -0.01 | -0.02 | wide - ffp |
| `sunny_day`, 250 Hz | -0.47 | -3.15 | -8.20 | -12.70 | ffp, dB |
| | -0.02 | -0.02 | +0.01 | +0.02 | narrow - ffp |
| | +0.01 | +0.02 | +0.02 | +0.03 | wide - ffp |
| `sunny_day`, 500 Hz | -7.51 | -13.17 | -22.26 | -30.27 | ffp, dB |
| | -0.09 | -0.10 | -0.10 | -0.09 | narrow - ffp |
| | +0.01 | +0.02 | +0.02 | +0.03 | wide - ffp |
| `clear_night`, 63 Hz | 5.73 | 5.97 | 6.41 | 6.78 | ffp, dB |
| | -0.05 | -0.04 | -0.01 | -0.00 | narrow - ffp |
| | +0.01 | +0.01 | +0.02 | +0.02 | wide - ffp |
| `clear_night`, 125 Hz | 4.39 | 4.37 | 4.18 | 3.78 | ffp, dB |
| | +0.02 | +0.03 | +0.03 | +0.03 | narrow - ffp |
| | +0.00 | +0.00 | +0.01 | +0.00 | wide - ffp |
| `clear_night`, 250 Hz | -1.06 | -3.81 | -10.28 | -17.94 | ffp, dB |
| | -0.07 | -0.07 | -0.13 | -0.28 | narrow - ffp |
| | -0.02 | -0.03 | -0.07 | -0.13 | wide - ffp |
| `clear_night`, 500 Hz | -8.15 | -11.48 | -13.57 | -13.55 | ffp, dB |
| | -0.08 | -0.07 | -0.04 | -0.04 | narrow - ffp |
| | -0.00 | -0.00 | -0.00 | +0.00 | wide - ffp |

The narrow-angle equation is within 0.11 dB of the program on average and 0.48 dB at
most (125 Hz, 300 m upwind in `windy`), the wide-angle one within 0.04 dB on average
and 0.18 dB at most. Over each band as the engine reads it, the mean energy over its
eight frequencies, the narrow-angle equation is within 0.08 dB on average and 0.38 dB
at most, the wide-angle one within 0.02 dB and 0.12 dB. In the weather near the
ground both hold, so the narrow-angle equation stays the default. In the
`open_air_stage` example the two equations differ by 0.15 dB at most, and the run
takes 13 s instead of 4 on four cores.

**Whole scenes.** The benchmark's cases 2 and 3 as whole scenes again (see
[Outdoor sound in wind](#outdoor-sound-in-wind-against-a-fast-field-program)), with
`parabolic_equation: wide_angle`:

**case 2, downward refraction 0.1/s**

| distance | 63 | 125 | 250 | 500 | 1k | |
|---|---|---|---|---|---|---|
| 100 m | 5.55 | 3.42 | -2.27 | -5.22 | 2.55 | ffp, dB |
| | +0.00 | +0.00 | -0.00 | -0.00 | +0.04 | engine - ffp |
| | +0.05 | +0.11 | +0.00 | -0.61 | +0.04 | through the filters |
| 200 m | 6.21 | 2.97 | -5.67 | -6.00 | 2.28 | ffp, dB |
| | +0.00 | +0.00 | -0.00 | -0.00 | -0.23 | engine - ffp |
| | +0.07 | +0.13 | -0.22 | -0.62 | -0.23 | through the filters |
| 500 m | 9.03 | 1.29 | -2.98 | -2.74 | 5.07 | ffp, dB |
| | +0.00 | -0.00 | -0.00 | -0.00 | -1.78 | engine - ffp |
| | +0.15 | -0.15 | -0.06 | -0.76 | -1.78 | through the filters |
| 1 km | 6.79 | 1.37 | 1.43 | 1.55 | 6.63 | ffp, dB |
| | -0.00 | +0.00 | -0.00 | -0.00 | +0.48 | engine - ffp |
| | -0.25 | -0.10 | +0.16 | -0.08 | +0.48 | through the filters |
| 2 km | 5.93 | 5.97 | -0.51 | 4.14 | 4.18 | ffp, dB |
| | -0.01 | -0.00 | -0.00 | -0.00 | -0.77 | engine - ffp |
| | -0.31 | +0.27 | -0.79 | +0.29 | -0.77 | through the filters |

**case 3, upward refraction 0.1/s**

| distance | 63 | 125 | 250 | 500 | 1k | |
|---|---|---|---|---|---|---|
| 100 m | 4.78 | 2.82 | -2.44 | -7.68 | -0.30 | ffp, dB |
| | -0.00 | -0.00 | +0.00 | +0.00 | -0.17 | engine - ffp |
| | +0.04 | +0.10 | +0.04 | -0.54 | -0.17 | through the filters |
| 200 m | 3.72 | 0.85 | -7.14 | -16.62 | -11.46 | ffp, dB |
| | -0.00 | -0.00 | +0.00 | +0.00 | -0.00 | engine - ffp |
| | +0.06 | +0.12 | -0.08 | -0.45 | -0.00 | through the filters |
| 500 m | -0.84 | -6.20 | -20.67 | -44.90 | -52.70 | ffp, dB |
| | -0.01 | -0.01 | -0.00 | +0.01 | +0.00 | engine - ffp |
| | +0.08 | +0.08 | -0.86 | -2.22 | +0.00 | through the filters |
| 1 km | -10.83 | -19.87 | -43.44 | -90.90 | -100.23 | ffp, dB |
| | -0.01 | -0.01 | +0.01 | - | - | engine - ffp |
| | +0.06 | -0.14 | -3.41 | - | - | through the filters |
| 2 km | -33.35 | -48.82 | -78.33 | -88.90 | -94.67 | ffp, dB |
| | -0.03 | -0.03 | - | - | - | engine - ffp |
| | -0.15 | -0.96 | - | - | - | through the filters |

In the bands the parabolic equation gives, the engine's levels are within 0.03 dB of
the program's band averages to 2 km, downwind and upwind, where with the narrow-angle
equation they are 0.5 to 0.8 dB off at 2 km downwind; upwind, in the shadow, where the
parabolic equation gives the 1 kHz band too, within 0.01 dB to 500 m, 53 dB down. The
1 kHz band downwind comes from the rays and does not change.

Each band of the wide-angle equation marches a grid of twenty points per wavelength of
its highest frequency in steps of three wavelengths, ten tridiagonal solves a step,
where the narrow-angle one takes a pair of Fourier transforms a step on a grid half as
fine, in steps of five wavelengths: three to four times the work, all of it in the
bands the parabolic equation gives. Neither equation is checked against measurements
in strong refraction, and neither is combined with a terrain yet (see
[Outdoor scenes](outdoor.md#what-is-not-modelled-yet)).

## Terrain against closed forms and the exact wedge

`validation/terrain.py` checks sound over a terrain (`geometry.ground.terrain`, see
[Outdoor scenes](outdoor.md#terrain)) against closed forms and the exact solution for a
wedge, and times it on a large one. The early sound reflects off the mean plane of the
ground and bends over its crests as the uniform theory of diffraction has it, wedge after
wedge, with the ground's reflections off the profile's segments on both sides of each
crest among its terms; the late rays bounce off the terrain's triangles. Levels are in dB
re the direct sound in free field over the same distance, the early paths added in
pressure in every band and the air's attenuation taken off each path over its own length.

**A flat terrain** (every height alike, 101 × 101 points 10 m apart) against the flat
ground: the direct sound and the one reflection have the same distances to the bit and
energies within 2e-16, and whole runs (20,000 rays, the same seed) the same level relative
to free field in every band, the two models differing only in how the ground is cut into
triangles.

**A plane sloping up 20 %** against the image source in the tilted plane: the reflected
path's distance is the image's to 2e-9 m and its energy within 1e-15 straight up the
slope, and within 2 mm and 7e-5 with the listener 160 m aside, where the ground also
tilts across the profile, which the mean plane, level across it as the standards have
it, leaves out.

**A ridge**, two slopes of 0.6 meeting at 241.9 degrees, with the source on one slope and
the listener in its shadow on the other, against the Biot-Tolstoy-Medwin solution for the
rigid wedge ([Edge diffraction](#edge-diffraction-against-the-exact-solution)), per
octave band; the engine's path is the wedge of the string pulled taut over the profile,
with the reflections off both slopes in its four terms:

| source, listener (m over the slopes) | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | |
|---|---|---|---|---|---|---|---|---|---|
| 2.0 at 30 m before, 1.5 at 30 m behind | -11.27 | -14.21 | -17.21 | -20.22 | -23.23 | -26.23 | -29.19 | -32.26 | exact, dB |
| | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 | -0.06 | +0.00 | engine - exact |
| 1.0 at 40 m before, 4.0 at 60 m behind | -13.31 | -16.28 | -19.28 | -22.29 | -25.30 | -28.11 | -31.50 | -34.26 | exact, dB |
| | -0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.21 | +0.18 | -0.07 | engine - exact |
| 3.0 at 20 m before, 1.5 at 40 m behind, 30 m aside | -9.82 | -12.73 | -15.72 | -18.72 | -21.73 | -24.77 | -27.73 | -30.73 | exact, dB |
| | +0.10 | +0.09 | +0.09 | +0.09 | +0.09 | +0.12 | +0.07 | +0.06 | engine - exact |

The engine agrees within 0.1 dB up to 1 kHz and 0.21 dB above, where the two slopes'
reflections interfere with the direct wave within a band: a ridge sampled on a grid is
the exact wedge, because the wedge sits at the crest with no width, its faces are the
slopes and the uniform theory's four terms are the exact solution's away from its
boundaries. The third pair is oblique, 30 m aside of the plane through the source and
the crest.

**A smooth hill**, 12 m high with a Gaussian profile 25 m wide at half its height, the
source and the listener 1.5 m up 40 m before and behind its crest, 2.71 m of path
difference over it. A rounded crest has no exact solution, so the sound over it is
compared with the engineering formulas the standards use for the same path
difference: over a soft ground (absorption 0.999), Maekawa's thin screen without
ground reflections (10 lg(3 + 20 N), ISO 9613-2 with C₂ = 40), and over a rigid
ground, ISO 9613-2's figure with C₂ = 20 (10 lg(3 + 10 N)), which stands for the
reflections off a typical ground:

| | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | |
|---|---|---|---|---|---|---|---|---|---|
| soft ground | -11.94 | -14.49 | -17.37 | -20.57 | -23.59 | -26.55 | -29.57 | -32.57 | engine, dB |
| | -13.60 | -16.28 | -19.14 | -22.07 | -25.04 | -28.03 | -31.03 | -34.03 | thin screen, 10 lg(3 + 20 N) |
| | +1.66 | +1.79 | +1.76 | +1.49 | +1.45 | +1.48 | +1.46 | +1.46 | engine - thin screen |
| rigid ground | -6.81 | -14.34 | -16.72 | -14.98 | -19.67 | -23.28 | -25.45 | -28.60 | engine, dB |
| | -11.12 | -13.57 | -16.28 | -19.14 | -22.07 | -25.04 | -28.03 | -31.03 | ISO 9613-2, 10 lg(3 + 10 N) |
| | +4.32 | -0.77 | -0.43 | +4.16 | +2.40 | +1.76 | +2.57 | +2.43 | engine - ISO |

Over the soft ground the engine lets 1.4 to 1.8 dB more over the hill than the thin
screen, evenly across the bands: the crest is the wedge of the hill's chords, open
wider than a half-plane, which bends more sound into its shadow. Over the rigid ground
the reflections off both flanks join the direct wave, each by way of the ground near
its foot, about 0.85 m longer than the straight line to the crest: in phase at 63 Hz
(5.1 dB more than over the soft ground, 4.3 above the standard's typical ground),
against it at 125 and 250 Hz, where that extra path is near half a wavelength (the
ground dip: 0.2 to 0.7 dB more than over the soft ground, a little below the
standard's figure), and from 500 Hz up 3.3 to 5.6 dB more than over the soft ground
(1.8 to 4.2 above the standard), where the waves no longer interfere alike across a
band and add in energy: 6 dB for four of a size, less because the reflected ones come
from images deeper in the shadow. The standard's figure stands for a ground that
reflects less than a rigid one. Both are engineering comparisons, not exact solutions.

**Climbing out of the shadow** behind the same hill over the rigid ground, from 1.5 to
29.5 m up in steps of 0.5 m, the level changes by at most 0.6 dB a step at 63 Hz and
3.0 dB at 8 kHz, where the edge of the shadow is sharpest (the line of sight clears the
hill from 23.2 m up), and never falls by more than 1.8 dB a step, in the ground's
interference and the lit zone's ripples:

| up (m) | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
|---|---|---|---|---|---|---|---|---|
| 1.5 | -6.81 | -14.34 | -16.72 | -14.98 | -19.67 | -23.28 | -25.45 | -28.60 |
| 4.0 | -9.64 | -14.02 | -16.65 | -16.40 | -19.95 | -23.27 | -25.98 | -29.13 |
| 6.5 | -9.86 | -11.74 | -15.29 | -15.95 | -19.10 | -22.40 | -25.13 | -28.20 |
| 9.0 | -8.00 | -11.19 | -14.31 | -15.17 | -18.31 | -21.72 | -24.51 | -27.59 |
| 11.5 | -6.44 | -10.57 | -13.44 | -14.17 | -17.38 | -20.80 | -23.59 | -26.66 |
| 14.0 | -5.21 | -9.91 | -12.21 | -12.93 | -16.04 | -19.42 | -22.22 | -25.29 |
| 16.5 | -4.36 | -8.72 | -10.61 | -11.22 | -14.06 | -17.34 | -20.11 | -23.17 |
| 19.0 | -3.66 | -7.49 | -8.57 | -8.92 | -11.12 | -13.96 | -16.55 | -19.52 |
| 21.5 | -2.85 | -6.24 | -6.14 | -6.06 | -6.95 | -8.40 | -9.65 | -11.44 |
| 24.0 | -2.05 | -5.02 | -3.53 | -2.94 | -2.20 | -1.46 | +0.32 | +1.17 |
| 26.5 | -1.18 | -3.93 | -1.16 | -0.31 | +0.85 | +0.96 | +0.93 | +0.55 |
| 29.0 | -0.37 | -3.25 | +0.79 | +0.79 | +0.24 | +0.99 | +0.46 | +0.90 |

At 63 Hz the level first falls as the listener rises, from -6.8 dB at 1.5 m to -9.9 at
6.5 m: the reflection off the ground behind the hill, in phase with the wave over the
crest at the ground, lags it more the higher the listener. From there every band rises
through the edge of the shadow, 12 dB in the 2.5 m round the line of sight at 8 kHz,
to within 1 dB of free field above it, where the direct sound, the ground's reflection
and the crest's wave ripple together.

**Cost.** Rolling hills 2 km across on a 20 m grid (10,000 cells, 21,233 triangles with
a building): the model builds in 3.9 s, its 8 diffracting edges (the building's) are found in
0.1 s, the early path over the hills in 0.3 s and a whole run with 20,000 rays takes
0.6 s, on four cores of the test container. The terrain costs the late rays nothing
beyond the triangles it adds, and the early sound the profile and the string over it,
once per listener.

## Outdoor scenes on the GPU

The GPU tracer (`simulation.late.device: gpu`, `both` or `auto`) follows the rays bent by
[wind and temperature](outdoor.md#wind-and-temperature), takes them through
[vegetation](outdoor.md#vegetation) and lets them out through the open sky, as the CPU's
does; the wave solver on the GPU steps its grid with the speed of sound per cell. The
tests check each against the CPU, and `validation/outdoor_gpu.py` compares whole runs of
the outdoor examples.

**Chords, ray for ray.** The GPU works in single precision, and its sines, cosines and
arc tangents are only guaranteed to about 11 bits, which moved a few percent of the
chords' ends by metres in a first port: the chords are therefore stepped without
trigonometry (the ray's direction kept as a cosine and sine pair, the turn over a step
summed as a series). 4,000 rays from random places in random directions, half of them
leaving a plane they must not meet again, in four atmospheres (sound speed rising or
falling 0.3/s to 100 m up, a wind of 6 m/s, and a wind of 4 m/s with the speed falling
0.1/s): the GPU's chords turn back where
the CPU's do (every one), point the same way at their ends to 3e-5, and stretch as far
to a part in 5,000 for 99 % of them (the rest end where the chord's tolerance is met a
little sooner or later as the turn gathered along it rounds, by up to 1.2 % where the
wind shears hard just above the ground). Cut to a common length, the two chords are the
same curve: ends within 0.32 mm and times within 4e-6 s, over 400 m.

**Energy and time.** In a field 400 m across under a 200 m sky, 200,000 rays per run,
listeners 100 m off downwind, upwind, across the wind and 60 m up: the energy per
listener and band and the mean time it arrives at, the rays bent up out of the field and
the energy over time the weight window reads, agree with the CPU's within the noise of
the ray count, in a warmer sky (sound bent down), a cooler one (bent up) and a wind with
a cooler sky. The energy over time is compared as what is left of it from each 10 ms
window on: in single precision a ray bent over many chords and bounces arrives a little
sooner or later now and then, and its energy lands in the next window (on every GPU, and
more on Direct3D 12's software GPU in Windows CI). With only mirror reflections in the
warmer sky, one window held 7 % less than the CPU's and the next 4 % more (15 % and 8 %
on Direct3D 12), while what is left from each window on agreed to 0.1 % (0.2 % on
Direct3D 12) wherever it was at least a hundredth of the whole: a shift the weight
window, rounded to quarter octaves, cannot tell. Through a crown of foliage and a stand
of stems, the energy each band loses, the energy scattered on and where it goes, with
and without an atmosphere, agree the same way; a tree belt of 17 media is declined with
its reason and traced on the CPU.

**Whole runs.** `validation/outdoor_gpu.py` simulates the outdoor examples with their own
settings once per device and once more on the CPU with the next seed, and compares the
late sound's level per band and its centre time at every listener (the mean absolute
difference over listeners and bands); the next seed shows how much the ray count's noise
alone moves them:

| Scene | Faces | Rays | Device | Late (s) | Level diff (dB) | Centre time diff (ms) |
|---|---:|---:|---|---:|---:|---:|
| open_air_stage | 12 | 50,000 | cpu, next seed | 0.59 | 4.21 | 6.0 |
| open_air_stage | 12 | 50,000 | gpu | 20.21 | 0.02 | 0.2 |
| tree_belt | 30 | 200,000 | cpu, next seed | 1.12 | 1.20 | 22.3 |
| tree_belt | 30 | 200,000 | gpu | 33.30 | 1.29 | 13.0 |
| hillside | 3,730 | 50,000 | cpu, next seed | 0.34 | 1.16 | 0.8 |
| hillside | 3,730 | 50,000 | gpu | 11.49 | 1.88 | 1.6 |

On SwiftShader, a software Vulkan driver on the container's 4 cores (the CPU runs took
0.4 to 1.0 s), which shows that the two agree, not how fast a GPU is; `python
validation/outdoor_gpu.py` times your own. The tree belt's rays through its foliage and
stems come out as far from the CPU's as the next seed's do. The stage in the wind
agrees to 0.02 dB: its late sound is the rain of a flat ground, which 50,000 rays sum
closely, while the next seed also redraws the eddies, which move the levels in the
wind's shadow by several dB on their own (the GPU run shares the CPU's eddies). On the
hillside the late sound at the listener on the slope is a handful of rays, so the mean
difference between CPU seeds ranges from 0.3 to 2.3 dB over six of them, and the GPU's
two seeds fall inside that range against every one (0.2 to 2.1 dB); the listener in
the shadow gets no late rays at all, and the one on the crest the same level from every
run within 0.02 dB.

**The wave solver in wind.** The GPU steps the tests' grids with a speed of sound that
grows with height, bending the sound as the rays do, to within 3e-5 of the largest
pressure of the CPU's (single-precision rounding; 2e-3 on Intel Macs, whose GPU drifts
further), its matched layer within 1e-5, and the 63 and 125 Hz bands of a scene whose
sound speed rises from 337 m/s at the ground to 402 m/s at the top of the grid within
1e-5, the two naming the same slowest and fastest speed of sound.

## Weather zones against a Crank-Nicholson parabolic equation

`validation/weather_zones.py` checks the engine where the weather changes on the way
(see [Weather zones](outdoor.md#weather-zones)). The reference is a Crank-Nicholson
parabolic equation written for the script (Salomons, *Computational atmospheric
acoustics*, appendix G): a narrow-angle equation marched in steps of a quarter
wavelength on a grid of a tenth of a wavelength, with the ground's impedance as its
lower boundary, a Gaussian starter and an absorbing layer at the top, which takes the
speed of sound of whichever zone each step lies in. The script first checks the
reference itself: against the exact solution for the reflection off grass in still air
it is within 0.17 dB at 63 to 500 Hz out to 800 m, and against the engine's own
parabolic equation in the `windy` preset over the whole way it is within 0.5 dB to 400
m, and 1.4 dB at 500 Hz 800 m out, where the two equations' starters and absorbing
layers part.

The source is 2 m high and the listener 1.5 m, over grass (flow resistivity 200 kPa
s/m²), and the weather changes 150 m from the source:

| case | from the source | past 150 m |
|---|---|---|
| shelter downwind | `windy` (5 m/s 10 m up, towards the listener) | 0.5 m/s |
| gust downwind | 0.5 m/s | 5 m/s |
| shelter upwind | 5 m/s towards the source | 0.5 m/s |
| inversion past a lake shore | `neutral` (still air) | `clear_night` (air warming with height) |
| cool water downwind | `windy` | from 100 to 300 m the ground at 12 °C instead of 20 |

**The parabolic equation through the zones.** The engine's parabolic equation marches
through the zones, changing its speed of sound where its path crosses a border. The
level re free field (dB) in pure tones, and in brackets the difference from the
reference of the engine's narrow-angle parabolic equation, its default, and then of
its wide-angle one (see
[The wide-angle parabolic equation](#the-wide-angle-parabolic-equation)):

| case, tone | 100 m | 200 m | 400 m | 800 m |
|---|---|---|---|---|
| shelter downwind, 63 Hz | 6.67 (-0.22, -0.07) | 7.14 (-0.10, -0.08) | 6.65 (-0.08, -0.06) | 5.49 (-0.08, -0.04) |
| shelter downwind, 125 Hz | 3.16 (+0.09, +0.03) | 0.97 (+0.14, +0.03) | -4.63 (+0.04, +0.06) | -15.71 (-0.22, +0.07) |
| shelter downwind, 250 Hz | -21.10 (-0.34, -0.02) | -22.24 (-0.17, +0.00) | -25.09 (-0.24, +0.02) | -28.48 (-0.38, -0.03) |
| shelter downwind, 500 Hz | -4.54 (+0.01, -0.01) | -4.37 (+0.13, +0.03) | -7.94 (+0.10, +0.08) | -9.24 (+0.08, +0.10) |
| gust downwind, 63 Hz | 5.50 (-0.02, -0.04) | 6.09 (-0.14, -0.04) | 7.35 (-0.14, -0.08) | 4.00 (-0.35, -0.11) |
| gust downwind, 125 Hz | 2.41 (+0.02, +0.02) | 1.11 (-0.04, +0.02) | -7.70 (+0.17, -0.09) | -12.62 (+0.03, +0.03) |
| gust downwind, 250 Hz | -10.56 (-0.04, +0.03) | -25.63 (-0.53, -0.09) | -23.87 (+0.39, +0.07) | -17.53 (+0.18, +0.02) |
| gust downwind, 500 Hz | -10.77 (-0.07, -0.01) | -11.59 (-0.12, +0.05) | -2.05 (-0.02, +0.05) | 1.88 (-0.36, -0.36) |
| shelter upwind, 63 Hz | 3.77 (+0.16, +0.05) | 2.59 (+0.01, +0.08) | 1.97 (+0.02, +0.05) | 0.45 (+0.02, +0.02) |
| shelter upwind, 125 Hz | 0.53 (-0.16, +0.07) | -2.60 (-0.26, +0.12) | -5.98 (-0.23, +0.09) | -12.88 (-0.20, +0.06) |
| shelter upwind, 250 Hz | -8.22 (-0.01, +0.07) | -17.69 (+0.08, +0.14) | -29.33 (+0.07, +0.12) | -39.75 (-0.12, -0.07) |
| shelter upwind, 500 Hz | -17.84 (+0.04, +0.20) | -35.34 (+0.02, +0.45) | -47.37 (-0.09, +0.48) | -61.17 (-0.14, +0.42) |
| inversion past a lake shore, 63 Hz | 5.36 (+0.00, -0.04) | 5.43 (-0.03, -0.04) | 5.71 (-0.01, -0.06) | 5.08 (-0.02, -0.07) |
| inversion past a lake shore, 125 Hz | 2.29 (+0.01, +0.02) | 0.69 (-0.00, +0.02) | -3.31 (+0.05, +0.01) | -16.52 (-0.12, -0.01) |
| inversion past a lake shore, 250 Hz | -10.04 (-0.02, +0.03) | -20.50 (-0.19, -0.00) | -27.08 (-0.01, +0.00) | -26.87 (-0.02, +0.05) |
| inversion past a lake shore, 500 Hz | -11.44 (-0.07, -0.00) | -15.67 (-0.11, +0.01) | -14.45 (-0.11, +0.01) | -8.31 (-0.07, +0.03) |
| cool water downwind, 63 Hz | 6.67 (-0.22, -0.07) | 7.70 (-0.23, -0.05) | 8.43 (-0.21, -0.05) | 4.36 (-0.42, -0.03) |
| cool water downwind, 125 Hz | 3.16 (+0.09, +0.03) | 1.03 (+0.18, +0.10) | -14.52 (+0.73, +0.32) | -15.33 (+0.22, +0.18) |
| cool water downwind, 250 Hz | -21.10 (-0.34, -0.02) | -24.13 (-0.18, +0.06) | -21.07 (+0.17, +0.07) | -15.33 (-0.01, -0.04) |
| cool water downwind, 500 Hz | -4.54 (+0.01, -0.01) | -0.95 (-0.07, +0.04) | 3.66 (+0.01, -0.03) | 0.89 (-1.19, -1.20) |

The engine is within 0.15 dB of the reference on average and 1.19 dB at most, at 500
Hz 800 m out over the cool water, where the two equations already part by 1.4 dB in
the uniform wind; the zones themselves add nothing to that. The wide-angle equation
is within 0.09 dB on average and 1.20 dB at most, at the same point, where the
engine's two equations agree within 0.01 dB and the reference, a narrow-angle
equation with a Gaussian starter, is the one that parts; elsewhere it is within
0.5 dB, the most 47 dB down in the upwind shadow at 500 Hz. The shelter downwind
bends the sound down over the first 150 m and then lets it spread, 13 dB less than
the uniform wind at 250 Hz 800 m out and 10 dB less at 500 Hz, where the wind would
bring the sound back to the ground; the gust turns still air's ground dip into the
wind's pattern past the border; the inversion past the shore carries 20 dB more than
still air at 500 Hz 800 m out; and the cool water deepens the 125 Hz dip 400 m out by
1.5 dB.

**The mean profile along the way.** The early paths bend through one profile, the mean
along their way, each zone's weighted by the length of the way over it. The parabolic
equation through that mean profile against the parabolic equation through the zones,
in the bands the rays give (500 Hz, 1 kHz and 2 kHz, averaged over the band's
frequencies), from 200 m on (nearer, the way lies under one weather); the level re free
field through the zones, and in brackets the mean profile's difference from it:

| case, band | 200 m | 400 m | 800 m |
|---|---|---|---|
| shelter downwind, 500 Hz | -4.54 (+0.01, -0.01) | -4.37 (+0.13, +0.03) | -7.94 (+0.10, +0.08) | -9.24 (+0.08, +0.10) |
| shelter downwind, 1 kHz | 3.11 (+1.38) | 3.87 (-0.72) | 4.06 (+1.75) |
| shelter downwind, 2 kHz | 3.45 (+1.84) | 1.96 (+3.60) | 0.33 (+6.57) |
| gust downwind, 500 Hz | -10.77 (-0.07, -0.01) | -11.59 (-0.12, +0.05) | -2.05 (-0.02, +0.05) | 1.88 (-0.36, -0.36) |
| gust downwind, 1 kHz | 3.22 (+1.28) | 2.24 (+1.25) | -0.88 (-2.71) |
| gust downwind, 2 kHz | 1.75 (+3.78) | -0.30 (+3.55) | -0.33 (-1.23) |
| shelter upwind, 500 Hz | -34.56 (+1.97) | -46.14 | -58.52 |
| shelter upwind, 1 kHz | -62.73 | -77.33 | -89.45 |
| shelter upwind, 2 kHz | -58.80 | -79.26 | -96.83 |
| inversion past a lake shore, 500 Hz | -11.44 (-0.07, -0.00) | -15.67 (-0.11, +0.01) | -14.45 (-0.11, +0.01) | -8.31 (-0.07, +0.03) |
| inversion past a lake shore, 1 kHz | 1.91 (+2.02) | 5.98 (+2.05) | 3.63 (+3.32) |
| inversion past a lake shore, 2 kHz | 3.67 (-0.13) | 5.80 (+1.65) | 4.04 (+2.48) |
| cool water downwind, 500 Hz | -4.54 (+0.01, -0.01) | -0.95 (-0.07, +0.04) | 3.66 (+0.01, -0.03) | 0.89 (-1.19, -1.20) |
| cool water downwind, 1 kHz | 3.47 (-0.06) | -0.66 (-0.01) | -2.39 (-0.10) |
| cool water downwind, 2 kHz | 3.11 (+0.18) | 1.89 (-0.34) | -1.10 (-0.11) |

Over the 37 points above -40 dB re free field the mean profile is within 1.6 dB on
average and 6.6 dB at most; the 8 points deeper in the upwind shadow are left out, as
sound there is inaudible next to any other and the parabolic equation itself is at
the floor of its absorbing layer. Where only the temperature changes on the way (the
cool water) the mean profile is within 0.3 dB. Where the wind changes it gives 1 to 4
dB more sound, 6.6 at most at 2 kHz 800 m downwind of the shelter: one profile cannot
both bend the sound down over the first 150 m and let it go straight after, so it
bends it a little the whole way and keeps the ground's reflection in step with the
direct sound further out. That is the cost of the equivalent profile the standards
take (Nord2000, Harmonoise), and the reason the bands up to 500 Hz, and up to 2 kHz in
a shadow, come from the parabolic equation, which goes through the zones.

## Levels over a long term

`validation/long_term.py` checks how the engine combines weather classes and the
periods of the day (see [Levels over a long term](outdoor.md#levels-over-a-long-term)).
The scene is a source 1 m over grass (Miki's impedance, 200 kPa s/m²) with the same
sound power in every octave band, and listeners 1.5 m up from 50 to 800 m away along
the wind, in four weather classes with shares of the day, the evening and the night as
a weather statistic might give them:

| class | weather | day | evening | night |
|---|---|---|---|---|
| downwind | `windy` (5 m/s 10 m up, towards the listeners) | 30 % | 25 % | 15 % |
| upwind | `windy`, towards the source | 30 % | 25 % | 15 % |
| still | `neutral` (still air) | 40 % | 30 % | 20 % |
| inversion | `clear_night` (air warming with height) | 0 % | 20 % | 50 % |

The downwind class is the scene's own weather, so the four classes take four runs,
about 20 s together. The level in each class, each period (12, 4 and 8 hours), over
the long term and L_den (5 dB in the evening, 10 at night), in dB(A):

| | downwind | upwind | still | inversion | day | evening | night | long term | L_den |
|---|---|---|---|---|---|---|---|---|---|
| 50 m | 46.5 | 36.6 | 47.4 | 47.7 | 45.6 | 46.1 | 46.8 | 46.1 | 53.0 |
| 100 m | 43.1 | 21.6 | 38.9 | 40.6 | 39.7 | 39.9 | 40.2 | 39.9 | 46.5 |
| 200 m | 33.6 | 10.5 | 26.3 | 36.1 | 29.3 | 31.8 | 33.9 | 31.8 | 39.8 |
| 400 m | 22.2 | -2.8 | 12.0 | 28.5 | 17.5 | 22.7 | 25.8 | 22.7 | 31.4 |
| 800 m | 12.2 | -20.2 | -1.6 | 17.7 | 7.2 | 12.0 | 15.0 | 12.0 | 20.7 |

**The arithmetic.** The script works the period levels, the long-term levels and L_den
out again from each class's own run: the energy mean over the classes by their shares
(ISO 1996-2, CNOSSOS-EU), over the periods by their hours, and L_den by Directive
2002/49/EC. The engine agrees with it to rounding (0.0 dB at most).

**The meteorological correction.** ISO 9613-2 predicts the level in conditions
favourable to propagation, downwind or under a moderate ground-based inversion such as
clear nights bring, and takes the long-term level from it as L_LT = L_DW - C_met, with
C_met = C_0 (1 - 10 (h_s + h_r) / d_p) beyond d_p = 10 (h_s + h_r) and 0 nearer, where
C_0 lies between 0 and 5 dB by the local weather statistics. The engine's C_met is the
A-weighted level in the favourable classes (downwind and the inversion, combined by
their shares of the whole day) less the long-term level. With C_0 fitted to it at 3.38
dB, within the standard's range:

| distance | engine | ISO 9613-2 | difference |
|---|---|---|---|
| 50 m | 0.94 | 1.69 | -0.76 |
| 100 m | 2.25 | 2.54 | -0.29 |
| 200 m | 3.13 | 2.96 | +0.17 |
| 400 m | 3.43 | 3.17 | +0.26 |
| 800 m | 3.49 | 3.28 | +0.21 |

The engine's correction follows the standard's shape within 0.76 dB, the most at 50 m,
where the standard's formula is only a step past d_p = 25 m and the ground's own
pattern still differs from one weather to the next. This is a check of plausibility,
not of accuracy: the standard's correction is an empirical fit, and the engine's comes
from each class's physics, which the sections above check against a fast field
program and parabolic equations.

## Moving sources against the exact field

`validation/doppler.py` checks walks with Doppler (see
[Moving sources and Doppler](spatial-audio.md#moving-sources-and-doppler)). A source
passes a listener 10 m from its line, from 60 m before it to 60 m past, 1 m up with the
listener 1.5 m up, over a rigid ground, at 30 m/s (a car) and 80 m/s (a train), and a
listener passes a source the same way at 30 m/s; the places are 1 m apart. Here the
field is known exactly: the direct sound and the image source's, each heard at its
retarded time (for a moving source the time t_e it left at, t = t_e + R(t_e) / c; for a
moving listener t - R(t) / c), over the distance then, with the air's absorption, for
a source that keeps its sound as it moves (quasi-static, as the engine's). Pure tones
through the engine are compared with it sample by sample (the waveform's error, the
energy of the difference relative to the field's) and in 20 ms windows (the level's
error, mean and most), away from the walk's ends:

| case | tone | waveform | level | without Doppler: waveform | level |
|---|---|---|---|---|---|
| car, 30 m/s | 250 Hz | -55.0 dB | 0.002 / 0.011 dB | 2.0 dB | 2.02 / 6.59 dB |
| | 1 kHz | -54.0 dB | 0.003 / 0.029 dB | 2.4 dB | 1.41 / 6.16 dB |
| | 4 kHz | -45.4 dB | 0.003 / 0.032 dB | 2.5 dB | 1.63 / 6.98 dB |
| train, 80 m/s | 250 Hz | -46.4 dB | 0.007 / 0.024 dB | 2.2 dB | 2.22 / 4.84 dB |
| | 1 kHz | -46.7 dB | 0.012 / 0.032 dB | 2.3 dB | 1.95 / 4.24 dB |
| | 4 kHz | -38.7 dB | 0.081 / 0.415 dB | 2.5 dB | 2.68 / 10.88 dB |
| passing listener, 30 m/s | 250 Hz | -55.0 dB | 0.002 / 0.011 dB | -2.4 dB | 2.01 / 6.05 dB |
| | 1 kHz | -54.2 dB | 0.003 / 0.019 dB | 1.1 dB | 1.29 / 5.79 dB |
| | 4 kHz | -45.5 dB | 0.003 / 0.019 dB | 2.4 dB | 1.53 / 6.11 dB |

With Doppler the sound follows the field to within 39 dB and its level to within 0.42
dB, the most for the train at 4 kHz, where its two paths cancel for a moment. The same
walks crossfaded from place to place, as without Doppler, keep each place's pitch, so
their phase jumps from place to place: the waveform has nothing in common with the
field's (an error of 0 dB or more) and the level misses by 4 to 11 dB at times.

**The sound sent each way.** The same walks with the source sending a tone each way
(an `Emission`, `acoustic_spaces.emission`): 1 + 0.8 cos θ in amplitude, θ from
straight ahead, and turned in phase by 90° times the sine of the elevation, on a grid
2° apart. In the exact field each path carries what is sent the way it leaves the
source, at the time it leaves:

| case | tone | waveform | level |
|---|---|---|---|
| car, 30 m/s | 1 kHz | -53.2 dB | 0.005 / 0.039 dB |
| | 4 kHz | -46.3 dB | 0.006 / 0.067 dB |
| train, 80 m/s | 1 kHz | -45.3 dB | 0.016 / 0.057 dB |
| | 4 kHz | -35.2 dB | 0.085 / 0.419 dB |
| passing listener, 30 m/s | 1 kHz | -53.5 dB | 0.005 / 0.047 dB |
| | 4 kHz | -46.1 dB | 0.007 / 0.105 dB |

With a cardioid directivity on the source, which the emission takes the place of in
the early paths, every number is the same to the digits shown. The level stays within
0.11 dB at 30 m/s and 0.42 dB for the train at 4 kHz, as one signal's does there; the
train's waveform at 4 kHz is 3.5 dB further than one signal's,
as the tone sent ahead weighs the approach more, where the pitch is higher and the
delays' error weighs more: sent louder behind instead (1 - 0.8 cos θ), it comes within
42.7 dB, and with the amplitude alike every way (only the phase turning) within 38.6
dB, as one signal. A grid 1° apart, or places 0.5 m apart, changes it by 0.3 dB at
most.

**The places.** The car at 1 kHz with its places further apart:

| places apart | waveform | level |
|---|---|---|
| 0.25 m | -57.1 dB | 0.001 / 0.016 dB |
| 1 m | -54.0 dB | 0.003 / 0.029 dB |
| 2 m | -38.5 dB | 0.008 / 0.055 dB |
| 4 m | -18.3 dB | 0.037 / 0.289 dB |

The delays are curves through the places, so the further apart the places, the more
they stray between them; up to 1 m they hardly do.

**A helicopter's places.** A source flying over a listener 150 m up at 63 m/s, from
300 m before it to 300 m past, over the rigid ground, the listener 1.2 m up right under
its track, as in a helicopter's flyover, with its places further apart (the level's
windows from 0.3 s after the first sound arrives):

| places apart | tone | waveform | level |
|---|---|---|---|
| 1 m | 250 Hz | -71.8 dB | 0.000 / 0.001 dB |
| | 1 kHz | -71.8 dB | 0.000 / 0.003 dB |
| | 4 kHz | -67.9 dB | 0.000 / 0.005 dB |
| 4 m | 250 Hz | -71.3 dB | 0.001 / 0.015 dB |
| | 1 kHz | -66.8 dB | 0.002 / 0.045 dB |
| | 4 kHz | -55.1 dB | 0.004 / 0.089 dB |
| 8 m | 250 Hz | -63.0 dB | 0.003 / 0.094 dB |
| | 1 kHz | -50.3 dB | 0.007 / 0.202 dB |
| | 4 kHz | -37.2 dB | 0.019 / 0.309 dB |
| 16 m | 250 Hz | -45.8 dB | 0.015 / 0.813 dB |
| | 1 kHz | -32.2 dB | 0.026 / 0.478 dB |
| | 4 kHz | -19.4 dB | 0.074 / 0.993 dB |

From 150 m up the delays curve far less than from a road 10 m away, so places 4 m
apart follow the field to within 55 dB at 4 kHz, more closely than the car's places 1 m
apart (45.4 dB at 4 kHz); the [flyover demo](demos.md#a-helicopter-flying-over)
simulates its helicopter every 4 m.

**The levels.** White noise through the car's walk over grass (Miki's impedance, 200
kPa s/m²), 16 noises up to 18 kHz: the energy of the sound in each octave band over the
pass-by, relative to the source's, against the noise through each place's own impulse
response for the time its sound is heard (Doppler moves the sound in time and pitch but
keeps its energy), and against the sound exposure level `pass_by` gives from the
places' band levels:

| band | rendered | responses | pass-by | early part alone: rendered | responses |
|---|---|---|---|---|---|
| 63 Hz | -25.88 | -25.90 | -26.13 | -25.90 | -25.92 |
| 125 Hz | -27.58 | -27.55 | -27.70 | -27.62 | -27.64 |
| 250 Hz | -32.16 | -32.10 | -32.30 | -32.31 | -32.31 |
| 500 Hz | -34.85 | -34.92 | -34.05 | -34.98 | -34.91 |
| 1 kHz | -29.52 | -29.53 | -29.64 | -29.56 | -29.59 |
| 2 kHz | -30.18 | -30.19 | -30.28 | -30.21 | -30.23 |
| 4 kHz | -30.76 | -30.77 | -30.68 | -30.78 | -30.79 |
| 8 kHz | -31.81 | -31.81 | -31.85 | -31.82 | -31.82 |

The sound keeps the energy of the responses to within 0.07 dB in every band, with its
late part and without. The late part is crossfaded as in any walk, drawn from one random
sequence for every place; the responses here are drawn from the same one, as otherwise
their late parts' chance correlation with the early sound would differ by up to 0.25
dB in a band. The pass-by's band levels are within 0.25 dB of the sound, 0.8 dB in the
500 Hz band, where the ground's dip falls: the same difference as between a place's
band levels and its impulse response filtered into bands. For a car at 30 m/s over
this grass with 94 dB of sound power, the listener has L_Amax 61.4 dB(A) 2.03 s into
the walk, SEL 61.2 dB(A) and L_Aeq 55.2 dB(A) over 4 s.

**What it leaves out.** The engine's source is quasi-static, as the field it is
compared with: a moving source's convective amplification (louder ahead of it by up to
0.8 dB at 30 m/s and 2 dB at 70 m/s, as much quieter behind) is left out, and so is the
Doppler shift of the late part, which is crossfaded. Binaural walks are checked
against the convolution with a response when the walker stands still (to within 45 dB
at each ear in the tests), not against a moving head's exact field.

## Helicopter rotors against theory and a certified level

`validation/rotorcraft.py` checks the sound a helicopter's rotors send (see
[Helicopters](outdoor.md#helicopters)): its tones against Gutin's formula and against
the far field summed over the blades in two other ways, the store that keeps them,
the airfoil self-noise against the sample outputs of its authors' code, the blades'
pitch against its closed form, Pegg's broadband noise against his worked example, and
the whole, with either broadband noise, against the certified noise level of the
helicopter it describes, the AS350 B3, flying at 63 m/s (25 °C).

**The tones.** Each harmonic of the blade passing frequency, against:

- Gutin's (1936) formula for a hovering rotor whose thrust and torque act on one
  radius, in 18 directions from the axis;
- the far field's integral over a turn, summed numerically with 4,096 instants a turn
  over the blades' points where they truly are, each a moving force (the lift and the
  drag) or a source (the thickness) on its straight chord, in 24 directions;
- Farassat's formulation 1A in time, the same points each heard at the time it left
  (solved by Newton's method) at an observer 20,000 km away in eight directions, its
  harmonics taken from 1,024 instants a turn. This check shares nothing with the
  closed form but the blade's loads.

| check | rotor | harmonics | most off |
|---|---|---|---|
| Gutin | 3 blades, 41.3 rad/s | 6 | 3 x 10⁻¹⁵ |
| the integral round the turn | main | 12 | 6.5 x 10⁻⁷ |
| | tail | 12 | 1.7 x 10⁻⁴ |
| in time (Farassat 1A) | main | 12 | 7.5 x 10⁻⁶ |
| | tail | 12 | 1.6 x 10⁻⁵ |

"Most off" is the largest difference of a harmonic's complex amplitude, over that
harmonic's loudest in any direction (over the loudest harmonic in the direction, in
time). The tail rotor's 1.7 x 10⁻⁴ comes from the chord's points' radius, taken to
second order: its chord is a fifth of its radius. In time the observer stands so far
away that the blade's radius squared over twice the distance, which the far field
leaves out of the phase, does not count: at 200 km it puts the main rotor 7.5 x 10⁻⁴
off, at 2,000 km 7.4 x 10⁻⁵, falling as one over the distance.

**The store.** A rotor's tones are kept as one period of the sound in each direction
on a grid (`Periodic`); what it sends over 4,000 samples 8.3 s in, each sample a
random direction, against the harmonics summed there:

| rotor | grid | harmonics | rms error | size | made in |
|---|---|---|---|---|---|
| main | 5° | 33 | -44 dB | 11 MB | 3.6 s |
| | 2.5° | 33 | -56 dB | 44 MB | 11.9 s |
| tail | 5° | 31 | -48 dB | 11 MB | 2.6 s |
| | 2.5° | 31 | -59 dB | 42 MB | 7.8 s |

The harmonics kept reach 90 dB under the loudest (`QUIET_DB`). The default grid is 5°
apart.

**Pegg's example.** Pegg's example I (NASA TM-80200, 1979), a hovering transport
helicopter heard 61.6 m from its main rotor, 5° under its disc, and 62.2 m from its
tail rotor, 10° off its axis: the broadband levels come out at 84.50 + S and 82.49 + S
dB (his 84.5 and 82.5), and the peak frequencies at 289 and 545 Hz (his 289 and 544).

**Airfoil self-noise.** Brooks, Pope and Marcolini's report (NASA RP-1218, 1989) gives
two sample outputs of its code: one untripped segment of a 30.48 cm chord at 1.516°
(table D2), and ten tripped segments of 15.24 cm at 5.4° with a rounded tip (D3), both
in a 71.3 m/s flow heard 1.22 m away. Each mechanism's band levels from 100 Hz to 40
kHz, against the tables:

| table | mechanism | most off |
|---|---|---|
| D2 | turbulent boundary layer: pressure side, suction side, angle of attack | 0.006, 0.006, 0.008 dB |
| | laminar boundary layer's vortex shedding | 0.003 dB |
| | total | 0.006 dB |
| D3 | turbulent boundary layer: pressure side, suction side, angle of attack | 0.006, 0.006, 0.006 dB |
| | tip vortex at 7.7° | 0.188 dB |
| | tip vortex at 7.668° | 0.006 dB |

The tables are rounded to 0.001 dB. The tip vortex column is the code's equations at a
tip angle of 7.668°, not at the 7.7° of the table's input; the code as the report
prints it gives 7.7°'s levels, 0.19 dB over the table at 100 Hz and 0.04 dB under it at
40 kHz, so the table may come from another version of it. Neither table has the
trailing edge's bluntness or a stalled section. For the AS350 B3 at 63 m/s, straight
down, the bluntness adds under 0.05 dB (its 0.5 mm trailing edges send theirs around
20 kHz) and the 1.5 % of the places round the turn where a section stalls nothing to
0.001 dB; the turbulent boundary layers give 106.8 of the main rotor's 107.5 dB 1 m
away, and the laminar boundary layer's vortex shedding 3 dB of the tail rotor's 89.4.
The directivity's angles are those of the report's figure B3: theta from the chord to
the observer, phi from the span to the observer's projection across the chord.

**The blades' pitch.** The collective and cyclic pitch that carry the trim's thrust
with no flapping, for the main rotor's rigid blades taken from the hub to the tip (no
root cut-out and no tip loss, so that it has a closed form), twisted by -10°, in an
even inflow, against Johnson's closed form (*Helicopter Theory*, 5.6):

| speed | theta_0 | theta_1s | most off |
|---|---|---|---|
| 0 m/s | 15.5675° | 0° | 9 x 10⁻¹² ° |
| 30 m/s | 13.2553° | -1.7615° | 7 x 10⁻¹² ° |
| 63 m/s | 14.5351° | -3.9524° | 2 x 10⁻¹¹ ° |

`tests/test_rotorcraft.py` checks the blade sections' sum (`bpm_bands`) against one
done place by place, each section's frame and motion through the air taken from the
disc's own vectors, each mechanism's level with its own directivity and its bands
moved one by one, hovering and at 63 m/s, to 10⁻⁹. Against 2,304 places round the
turn, the default 144 hold the main rotor's level, on the 5° grid, to 0.01 dB
everywhere below the horizon and to 0.7 dB above it, close to its disc's plane. The
tail rotor's disc is upright, so listeners below the helicopter are close to its
plane: there its level holds to 1.3 dB (0.6 dB with 288 places, 0.3 dB with 576),
because BPM's directivity changes abruptly where a section's chord points at the
listener, an angle where the report itself finds it inaccurate. The tail rotor's
broadband noise is 36 dB under the helicopter's sound in the certification's overflight
(below). 48 sections instead of 24 change the levels by under 0.06 dB.

**The certified level.** The AS350 B3 at its maximum mass of 2,250 kg flying the
overflight of ICAO Annex 16, chapter 11 (14 CFR part 36, appendix J): level, 150 m
over a microphone 1.2 m above grass, at 0.9 V_H, in still air at 25 °C and 70 %
humidity at sea level. Its type certificate data sheet for noise (EASA.R.008) gives
no speed; V_H is taken as Airbus's fast cruise speed, 136 kt, so 63.0 m/s. The direct
sound and the ground's reflection (Weyl and van der Pol's, with Miki's impedance of
200 kPa s/m²), each heard at the time it left the helicopter with the air's absorption
(ISO 9613-1), A-weighted; the slow time weighting finds the points 10 dB under the
most, and the sound exposure level is the energy between them:

| part | SEL | L_ASmax | 10 dB down for |
|---|---|---|---|
| all | 88.0 dB(A) | 81.7 dB(A) | 9.4 s |
| main rotor's tones | 63.3 dB(A) | 55.6 dB(A) | 8.9 s |
| tail rotor's tones | 79.8 dB(A) | 72.4 dB(A) | 10.8 s |
| main rotor's broadband noise | 87.3 dB(A) | 81.5 dB(A) | 7.9 s |
| tail rotor's broadband noise | 64.9 dB(A) | 57.8 dB(A) | 11.0 s |
| certified | 84.6 dB(A) | | |

That is with Pegg's broadband noise, the default. With each blade section's own
(`broadband="bpm"`):

| part | SEL | L_ASmax | 10 dB down for |
|---|---|---|---|
| all | 80.1 dB(A) | 72.4 dB(A) | 12.0 s |
| main rotor's broadband noise | 68.6 dB(A) | 62.2 dB(A) | 8.4 s |
| tail rotor's broadband noise | 44.3 dB(A) | 37.4 dB(A) | 10.2 s |

With Pegg's noise the model is 3.4 dB louder than the certified level, nearly all of
it Pegg's broadband noise of the main rotor; with the blade sections' own it is 4.5
dB quieter, nearly all of it the tail rotor's tones, the main rotor's broadband noise
19 dB under Pegg's. Self-noise is the noise of a blade in a smooth flow: it leaves out
the blades cutting through the wake of those ahead and through turbulence, which a
full-size rotor in flight makes and which Pegg's fit to measured rotors holds. Pegg's
method is a correlation of the 1970s' measurements: in his own example the band of the
broadband peak (315 Hz) comes out 5.3 dB over the level measured there, though a tone
of the tail rotor in that band is over it alone; the bands where only the broadband
noise counts have no measured level in the report. The tones leave out the loads that
change round the turn, which make a rotor's lower harmonics in forward flight louder
than steady loads do. Some of the AS350 B3's inputs are assumed (the main blades' chord
and twist, the tail rotor's speed, chord and thickness, V_H); `AS350_B3`'s source code
says which.

## Edge diffraction in whole cities

Edge diffraction looks, for the source and for each listener, at every pair of an edge
and a plane they face, for a path that reflects off the plane before or after bending
over the edge. In a city the edges and the planes both grow with the buildings, so the
pairs grow with their square: 900 buildings made 20 million pairs per listener, and the
engine ran out of memory from about 1,600 buildings. Most pairs cannot carry a path.
The ray tests start just off the edge and head for the mirror image of the source or
listener in the plane, so a ray from an edge behind the plane never reaches it, and one
from the rest of the edge meets the plane where the image casts the edge's shadow on
it, to within a margin that the ray's slant sets. Where that shadow misses the box
round the plane's faces, no reflection can happen, and the engine leaves the pair out
before tracing a ray. For a listener's images it also leaves out the paths that would
touch the edge's line beyond its ends or arrive too late. Pairs whose rays graze the
plane, or whose image lies too near it for the margin to hold, are all kept. The exact
tests then run on what is left, as before, so the results do not change.

`validation/city_speed.py` builds cities of n by n blocks 32 m apart, each a building
of random size (12 to 20 m a side) and height (6 to 30 m), with grass in the streets.
A source stands in a street crossing in the middle, 1 m up, and two listeners stand 3
and 5 blocks off. The simulation is the outdoor default: early reflections to the third
order, the direct sound and first-order reflections bending over the edges, and 50,000
rays per band for the late part. Times are on 4 cores of a cloud machine:

| Buildings | Faces | Edges | Build (s) | Simulation (s) | Of it diffraction (s) | Pairs tested | Peak memory | Before (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1,012 | 800 | 0.2 | 3.7 | 3.0 | 8.5 % | 0.2 GB | 1.3 |
| 400 | 4,012 | 3,200 | 1.1 | 9.1 | 8.4 | 4.3 % | 0.3 GB | 15.1 |
| 900 | 9,012 | 7,200 | 2.1 | 16.6 | 15.6 | 2.9 % | 0.3 GB | 81.1 (7.1 GB) |
| 1,600 | 16,012 | 12,800 | 4.2 | 21.4 | 20.1 | 2.7 % | 0.5 GB | out of 14 GB |
| 3,600 | 36,012 | 28,800 | 10.2 | 56.4 | 54.5 | 3.3 % | 1.7 GB | |

Diffraction includes the terms that make the other specular paths fade at the edges
([above](#reflections-that-fade-at-edges-against-the-exact-solution)): without them,
900 buildings simulate in 2.6 s (1.6 s of it diffraction), 1,600 in 4.8 s (3.6 s) and
3,600 in 24.4 s (22.4 s).
"Before" is the same run with every pair tested and without those terms, as in
earlier versions. The tested
share stays near 3 %: the shadow of a wall facing the source covers a wedge of the city
that widens away from it, and only the ray tests tell whether other buildings hide the
wall. The work therefore still grows with the square of the buildings, and a few
thousand buildings take tens of seconds per source. Tens of thousands need the map cut
down to the area round the sources and listeners, or `diffraction: direct`, whose cost
grows with the edges alone.

**Exactness.** The script's check runs a city of 4 by 4 blocks with the buildings
turned at random angles and paths that reflect on both sides of an edge. It runs two
sources, one in a street and one over the roofs, to two listeners, once as the engine
runs, on one thread and on every core, and once with every pair tested. Every path, its
energy and its spectrum come out the same bit for bit; the early paths took 1.1 s
instead of 4.3 s. The same comparison over every example and three test cities, 468
source and listener pairs with paths reflecting on one side and on both sides of an
edge, and over two edges, found no difference. The tests check the culling against
rays traced from random edges to a tilted rectangle of faces, at grazing angles, for
faces that stray from their plane and for planes with only openings.

## Wave-based solver

`validation/wave_solver.py` checks the wave solver against theory: closed forms where
there are any, and for a whole room the sum of its modes. It has two stencils, each at
its stability limit: the standard one (7 points, the six axial neighbours) and the
interpolated isotropic one (19 points, the twelve edge neighbours too), which closed
rooms take. Each check runs with both where both apply. (Each takes its Laplacian
times 1 − 2⁻¹⁸, which slows sound by 2⁻¹⁹: at the limit itself the highest modes, the
7-point stencil's checkerboard and the 19-point one's modes alternating along two
axes, sit on the edge of stability, where a kick grows linearly and single precision's
rounding can take them over it. 2⁻¹⁶ inside it, they swing within 259 times a kick
over 300,000 steps in single precision.)

**Speed of sound on the grid.** Waves on the grid travel a little slower the shorter
they are. From the peaks of a rigid 2.4 x 2.4 x 0.2 m box (10 cm cells) ringing for
8 s, against the exact frequencies of its modes and against each stencil's dispersion
relation, 2 − 2 cos ωT = Σ g_k (1 − cos k·d_k) over its links:

| direction | mode | cells per wavelength | 7-point, relation | wave solver | 19-point, relation | wave solver |
|---|---|---:|---:|---:|---:|---:|
| along an axis | (6, 0, 0) | 8.00 | −1.740 % | −1.740 % | −0.672 % | −0.672 % |
| along a diagonal | (4, 4, 0) | 8.49 | −0.387 % | −0.388 % | −0.573 % | −0.573 % |

Both follow their relations within 2e-5. At 8 cells per wavelength the 7-point stencil
is 1.74 % slow along the axes, 0.44 % along the face diagonals and exact along the body
diagonals; the 19-point one is 0.65 to 0.67 % slow in every direction. It needs 5.2
cells per wavelength to be as close as the 7-point one is at 8.

**Modes and their decay.** A 7 x 5 x 3 m box rings for 8 s (10 cm cells). Its walls'
admittance changes with frequency as a material's does on the wave solver's grid: a
constant 0.002 and a branch resonating at 60 Hz (Q 1.5, admittance 0.008 at resonance),
so below 60 Hz they give like a spring and above it weigh like a mass. For each mode
that stands apart from the others, the frequency of its peak and the decay rate of its
envelope match first-order perturbation theory: the decay rate δ = c Re(β)/2 Σ (4/L
along each axis the mode varies, 2/L along the others), β the walls' admittance at the
mode, and the rigid box's frequency moved by −c Im(β)/(4π) times the same sum. With
either stencil the frequencies come out within 0.25 % of theory (the walls move them by
up to 0.6 %), the decay rates, which differ fivefold, within 1.7 %:

| mode | rigid box (Hz) | theory (Hz) | 7-point (Hz) | 19-point (Hz) | Re(β) | decay, theory (1/s) | 7-point (1/s) | 19-point (1/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (1, 0, 0) | 24.50 | 24.39 | 24.39 | 24.39 | 0.0028 | 0.779 | 0.770 | 0.770 |
| (0, 1, 0) | 34.30 | 34.14 | 34.14 | 34.15 | 0.0039 | 1.185 | 1.167 | 1.168 |
| (1, 1, 0) | 42.15 | 41.93 | 41.91 | 41.91 | 0.0057 | 1.990 | 1.955 | 1.958 |
| (2, 0, 0) | 49.00 | 48.84 | 48.87 | 48.88 | 0.0078 | 2.200 | 2.191 | 2.195 |
| (0, 0, 1) | 57.17 | 57.10 | 56.97 | 56.98 | 0.0098 | 3.407 | 3.442 | 3.445 |
| (0, 2, 1) | 89.30 | 89.56 | 89.55 | 89.54 | 0.0052 | 2.152 | 2.124 | 2.117 |

Each decay rate is the slope of the mode's envelope from 1 s on, over 50 dB at most:
the fastest modes reach the noise floor of single precision before the run ends. The
theory's rates move by up to 0.35 % on the 19-point grid, whose time step is longer.

Axial modes lose energy at walls they meet head-on and at walls they graze, so this
checks absorption at all angles.

**Walls.** A plane wave down a duct (10 cm cells) reflects off its end as
(1 − β)/(1 + β) predicts at normal incidence, and grazing its absorbing walls it decays
as exp(−βP/(2A)) per metre (perimeter P, cross-section A), whatever the cell size:

| end admittance β | 0 | 0.1 | 0.25 | 0.5 | 1 |
|---|---:|---:|---:|---:|---:|
| reflection, theory | 1.000 | 0.818 | 0.600 | 0.333 | 0 |
| 7-point stencil (60 to 150 Hz) | 0.998 | 0.817 | 0.601 | 0.336 | 0.044 |
| 19-point stencil | 0.999 | 0.818 | 0.602 | 0.337 | 0.044 |

| walls β = 0.05, 0.6 x 0.6 m duct, amplitude after 10 m | theory | 10 cm cells | 5 cm cells |
|---|---:|---:|---:|
| 7-point stencil | 0.1889 | 0.1900 | 0.1906 |
| 19-point stencil | 0.1889 | 0.1906 | 0.1907 |

**Walls that absorb as their materials do.** The wave solver gives each material an
admittance that changes with frequency: a constant and resonant mass-spring-damper
branches, one at each wave band's centre and one beyond each end, fitted so that the
random-incidence absorption, counted through the analysis's band filters, matches the
material's in every band. At the grid of the 500 Hz band:

| material, 63 to 500 Hz | absorption | fitted | branches |
|---|---|---|---:|
| porous absorber | 0.05, 0.15, 0.40, 0.70 | 0.050, 0.150, 0.400, 0.700 | 6 |
| panel absorber | 0.35, 0.30, 0.15, 0.08 | 0.350, 0.300, 0.150, 0.080 | 5 |
| audience | 0.50, 0.60, 0.75, 0.85 | 0.500, 0.600, 0.750, 0.850 | 6 |
| plaster on concrete | 0.01, 0.01, 0.015, 0.02 | 0.010, 0.010, 0.015, 0.020 | 6 |
| a steep resonance | 0.10, 0.60, 0.25, 0.10 | 0.136, 0.579, 0.253, 0.100 | 4 |

The octave filters overlap, so a band next to a much more absorbing one cannot absorb
much less than it lets through: the steep resonance is met within 0.04. Simulations
report the largest miss as `wave_absorption_miss`. With the porous absorber's fitted
admittance at the end of a duct (5 cm cells, 0.2 m across), against a rigid end's
reflection, which the grid returns whole at every frequency:

| frequency (Hz) | 31.5 | 44 | 63 | 88 | 125 | 177 | 250 | 354 | 500 | 707 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7-point: normal-incidence absorption, fitted admittance | 0.030 | 0.022 | 0.022 | 0.037 | 0.075 | 0.138 | 0.259 | 0.366 | 0.567 | 0.597 |
| 7-point: the grid's theory | 0.030 | 0.022 | 0.022 | 0.037 | 0.074 | 0.137 | 0.253 | 0.355 | 0.542 | 0.571 |
| 7-point: wave solver | 0.030 | 0.022 | 0.022 | 0.037 | 0.075 | 0.136 | 0.254 | 0.353 | 0.540 | 0.566 |
| 19-point: normal-incidence absorption, fitted admittance | 0.030 | 0.022 | 0.022 | 0.037 | 0.072 | 0.143 | 0.268 | 0.346 | 0.532 | 0.659 |
| 19-point: the grid's theory | 0.030 | 0.022 | 0.022 | 0.036 | 0.072 | 0.142 | 0.262 | 0.338 | 0.509 | 0.651 |
| 19-point: wave solver | 0.030 | 0.022 | 0.022 | 0.037 | 0.072 | 0.142 | 0.263 | 0.339 | 0.509 | 0.651 |

(Each stencil has its own time step, so its own fit.) The wave solver matches the
grid's theory (a plane wave on the grid, the wall half a cell past the last cell
centres) within 0.005 with the 7-point stencil and within 0.001 with the 19-point one,
from 30 to 720 Hz. That half cell takes a little of the absorption away as the
wavelength shortens: within 0.015 (7-point) and 0.009 (19-point) of the fitted
admittance's below 360 Hz, 0.027 and 0.023 less at 500 Hz with 14 cells per
wavelength. The fit does not correct for it.

**Open models.** Round a model whose air reaches the edge of the grid, the grid grows
by a perfectly matched layer (split-field, its damping growing as the square of the
depth). A rigid 8 x 6 m plate in the open, the source 1.5 m above it, three listeners
round it, 25 to 250 Hz (20 cm cells, the 7-point stencil, the only one with the
matched layer), against a grid 16 m larger all round, from whose
edge nothing comes back in the 90 ms of the runs: what differs, relative to the sound
itself,

| boundary of the grid | listener 1 (dB) | listener 2 | listener 3 | air cells |
|---|---:|---:|---:|---:|
| the grid's edge alone (admittance 1, as in 2.0.0) | −6.5 | −13.5 | −11.0 | 28,380 |
| matched layer, 6 cells | −60.2 | −64.8 | −56.6 | 79,200 |
| matched layer, 8 cells (the default) | −66.6 | −74.0 | −69.7 | 104,076 |
| matched layer, 12 cells | −73.6 | −77.0 | −62.4 | 168,036 |

The edge alone absorbs only sound that meets it head-on. In the layer the sound keeps
falling over 59,409 steps (20 s): 79 dB below the first second's level after one second,
175 dB below after 18.

**One run for all bands.** A 7 x 5 x 3 m box whose absorption differs in every band
(0.10 to 0.25 from 63 to 500 Hz), the wave solver up to the 500 Hz band: one run of
868,701 cells and 3,997 steps of the 19-point stencil, 7.8 s on 4 cores with the fit and
the grid's set-up, where 2.0.0 ran one per band. The run itself takes 7.0 s, against
5.0 s for the same grid with constant walls: the branches cost about 40 % more than a
run with constant walls, and save three runs.

**Free field.** In a 30 x 24 x 24 m room with 95 % absorption, the direct sound's
energy in the 63 and 125 Hz bands at 1 to 8 m is within 0.2 dB of the free-field level
the rays give with the 19-point stencil (−0.07 to +0.04 dB at 63 Hz, +0.05 to +0.20 dB
at 125 Hz), and within 0.35 dB with the 7-point one (−0.05 to −0.13 dB at 63 Hz, −0.16
to −0.33 dB at 125 Hz).

**A whole room.** A 7 x 5 x 3 m box with 15 % absorption on every surface, simulated
with the wave solver switched on at its default settings (the 63 and 125 Hz bands,
20 cm cells, the 19-point stencil),
against the sum of the box's modes up to 1.5 kHz, each with its decay rate, all
analysed alike (ISO 3382-1, octave filters):

| model | G 63 Hz (dB) | G 125 Hz | EDT 63 Hz (s) | EDT 125 Hz | T20 63 Hz (s) | T20 125 Hz |
|---|---:|---:|---:|---:|---:|---:|
| sum of the modes | 20.3 | 23.5 | 1.02 | 0.96 | 0.94 | 0.90 |
| wave solver | 21.1 | 22.9 | 1.04 | 0.95 | 0.94 | 0.87 |
| wave solver, 7-point stencil | 21.1 | 22.9 | 1.04 | 0.93 | 0.94 | 0.86 |
| wave solver, 16 cells per wavelength | 20.8 | 23.7 | 1.01 | 0.84 | 0.90 | 0.90 |
| rays only | 25.6 | 21.8 | 0.71 | 0.77 | 0.91 | 0.79 |

![Transfer function of the box from the sum of its modes, the wave solver and rays](images/wave.png)

The wave solver follows the modes peak for peak from 22 Hz up. What is left comes
mostly from where the grid puts the walls: within half a cell of the model's, here
6.93 x 4.95 x 2.97 m with 20 cm cells, which moves each mode by up to 1 %. In a room
this sparse in modes, that moves G by up to 1 dB, EDT by up to 12 % and T20 by up to
5 %; more cells per wavelength move the walls but do not make every figure converge.
The two stencils agree within 0.02 s: in this band their dispersion moves the modes
less than the walls' placement does. Rays give the average over the room instead: 5.3 dB more at this seat in the 63 Hz
band, where the listener sits near the nodes of most modes.

**On a GPU.** `validation/wave_gpu.py` computes the wave bands (up to 125 Hz, 0.5 s) of
the first source of three example rooms to every listener on the CPU and on the GPU,
and compares the two. The tests do the same with resonant walls and with the matched
layer, in runs split so that the fields swap both ways between them, with both
stencils. On Mesa's lavapipe, a Vulkan driver that runs on the CPU, with 4 cores (the
19-point stencil):

| room | cells | steps | CPU (s) | GPU (s) | largest difference |
|---|---:|---:|---:|---:|---:|
| studio | 10,350 | 1,016 | 0.2 | 1.5 | 2.8e-7 |
| concert hall | 1,223,312 | 1,016 | 2.9 | 25.4 | 6.8e-7 |
| church | 1,228,682 | 1,016 | 3.5 | 30.1 | 1.5e-6 |

The difference is relative to the largest pressure: the GPU's single-precision
rounding, which stays at that level. Apple's GPU, in CI, rounds further from the CPU:
in the test's nearly closed room with resonant walls the two differ by up to 1.1e-5
after 1,601 steps, in the slow drift such a room builds up. These times are from an
earlier version of the script and include the grid's set-up on the CPU (the same on
both) and, in the first room, the GPU's start-up; the script now prints the set-up on
its own, the CPU's and the GPU's time steps without it, and the speed-up of the steps.
A software driver is no faster than the CPU solver; the script is there to time a
hardware GPU.

## Real-time convolver

The plan's target: a 3 s response with 16 output channels (third-order Ambisonics), fed
by one input in blocks of 128 frames at 48 kHz, without dropouts for 10 minutes. The
benchmark sends 10 minutes of noise (225,000 blocks) through the convolver from Python,
as the playback callback does, on one core of a shared 4-core x86-64 cloud machine, and
times every block:

| per block | mean | median | 99 % | 99.9 % | max | over the 2.667 ms budget |
|---|---:|---:|---:|---:|---:|---:|
| thread CPU time | 0.298 ms | 0.239 ms | 0.736 ms | 0.903 ms | 2.055 ms | 0 of 225,000 |
| wall-clock time | 0.304 ms | 0.241 ms | 0.763 ms | 1.531 ms | 6.874 ms | 48 of 225,000 |

The response is split into 16 partitions of 128 samples, 14 of 1024 and 16 of 8192, and
the work of the long partitions is spread over the blocks in between. The convolver's
own work (CPU time) never took longer than a block lasts: on average 11 % of it, at most
77 %. In wall-clock time, 48 blocks (0.02 %) were late because the shared machine took
the core away; on a desktop with a real-time audio thread and the usual two or more
buffers at the sound card, such single late blocks are absorbed. Whether a given sound
card plays 10 minutes without a dropout can only be checked on that machine:
`acoustic-spaces play ir.wav --live --block 128 --seconds 600` reports the longest time a block
took and every block the sound card dropped. The output matched offline convolution to 2.6 × 10⁻⁷ of
its peak (single precision).

## Known issues

A review of the code on 25 October 2026 found these, but for the two about barriers,
which the checks against the exact solution found. Each one was checked against the
code, and the numbers come from probes or from the code's own formulas; they are open
items on the [roadmap](roadmap.md).
The edge fade's own are listed in
[Known issues of the edge fade](#known-issues-of-the-edge-fade).

Results:

- **The first milliseconds of a rendered response.** An impulse response is put
  together from its bands with zero-phase filters (`synthesis.combine_bands`), which
  spread each arrival both ways in time, and what would fall before the response's
  first sample is lost. For an arrival whose level differs from band to band, that
  changes its level in the low bands when it comes in the first milliseconds: the
  direct sound of a source whose level towards the listener is 1.5 dB lower at 125 Hz
  than at 63 Hz, 4.7 dB at 250 Hz and 20 dB from 2 kHz up has a G 0.33 dB lower at
  63 Hz and 0.55 dB higher at 125 Hz at 1 m (2.9 ms in), and 0.31 dB lower and 0.15 dB
  higher at 2 m, than the same sound 200 ms later; an omnidirectional source's direct
  sound changes by 0.014 dB or less. The echograms, and the parameters `simulate`
  prints and writes from them, are not affected; the parameters `analyze` takes from
  the WAV files are.
- **The early order with vector-based scattering.** Vector-based scattering (the
  default) sends each reflection along the sum of its specular direction and a Lambert
  direction, which leans towards the wall's normal, so the late part carries the
  specular share of the first orders along other paths than the exact ones, and with
  more energy at the listener: in a 4 x 3.5 x 3 m box (absorption 0.15, scattering
  0.3), a quarter more for orders 1 to 3 near the middle of the room, and 4 to 53 % more
  over four positions and scattering from 0.1 to 0.6. The early part replaces those
  orders with the exact paths, so the total depends a little on how many it takes:
  with the default `early.max_order: 3` it is within 0.04 dB of taking orders 1 to 9,
  and with `max_order: 0` 0.15 to 0.2 dB higher (from 250 Hz up). Random scattering
  keeps the specular share on the specular paths: within 0.03 dB either way
  (`tests/test_scattering.py`). A lobe centred on the specular direction would remove
  this, but changes the late part's decay, which the default was chosen to match in
  the BRAS rooms.
- **Pairs of paths from 1 kHz up.** In the bands above 500 Hz that add in pressure
  (outdoors, 1 to 8 kHz), the echogram shares each pair of paths' cross term between
  their two time slots, and a slot that would go below zero is set to zero
  (`coherent.pair_energy`). That adds energy where paths cancel. Over all 419 pairs of
  a source and a listener in the nine outdoor examples, it adds at most 0.009 dB to a
  band's response (the `street`'s garden at 1 kHz) and nothing elsewhere; to the
  lateral figures of eight behind LF, LFC and LG, at most 0.04 dB (the `street`'s
  balcony at 1 kHz) and 0.003 dB elsewhere. Impulse responses add the paths in pressure
  without it.
- **A barrier on a hard ground with `first_order`.** The path over the barrier's top
  with the reflection off the ground behind it is left out where the reflection off the
  ground before the barrier is found: both come from the source's image in the ground,
  and the edge diffraction takes that reflection for the geometric wave of both. 4 m
  behind a barrier 3 m high the level at 63 Hz steps by 0.35 dB where the listener
  rises into sight of that reflection, and is 0.33 dB low on average above it (0.07 dB
  or less from 125 Hz up; see
  [Reflections that fade at edges](#reflections-that-fade-at-edges-against-the-exact-solution)).
  A fix that tells the two paths apart removes the step, but changes the early level of
  BRAS CR2 by up to 0.8 dB at 63 Hz, where 12 terms over its 10 pairs of a source and a
  listener that were left out count with it (the edge fade's terms stay the same); it
  waits for a run of BRAS with it.
- **Paths over two edges take no reflections.** With `second_order` a path over two
  edges reflects nowhere before, between or after them, so behind a thick barrier on a
  hard ground the engine has one of the four paths over it: 1.7 to 5.7 dB low from
  500 Hz up, and off by up to 9.9 dB below (see
  [Reflections that fade at edges](#reflections-that-fade-at-edges-against-the-exact-solution)).

Models:

- **Collada polygons with holes** (`<ph>`) are read with their holes filled.

Files and runs:

- **No limit on time.** The numbers of rays have no upper bound: a mistyped count runs
  as long as it asks for. (The memory of a run's late part is estimated before it
  starts, and a run whose late part would need more than the computer has is refused.
  The early stage's is not estimated, and in a room of many facets with edge
  diffraction it is most of a run's: BRAS CR3 peaks at 10.0 GiB in its edge fade
  (11.7 GiB in third-octave bands), which in third-octave bands starts each listener
  from less than 0.9 GiB; see [Memory](simulation.md#memory).)
- **No uncertainty.** A run gives no standard errors: nothing says how much its
  parameters would change with another seed or more rays. Only `calibrate` gives one,
  for the absorption it fits.

## Not validated yet

- **More measured rooms.** Two rooms have been compared with measurements, by their
  ISO 3382-1 parameters. G, the spatial outputs and the parameters that need directions
  (LF, IACC, stage support, directional decay; BRAS also has binaural responses) and
  BRAS's auditorium, CR4, whose SketchUp file no free reader opens, have not.
- **Wave effects.** Above the wave solver's bands the simulation is geometric. Edge
  diffraction is modelled (UTD, for the direct sound and first-order reflections, and
  the direct sound over two edges; the fade of every other specular path at edges,
  which leaves some steps, see
  [Known issues of the edge fade](#known-issues-of-the-edge-fade)) and room modes by
  the wave solver, both checked against theory but not against measurements in rooms. Without the wave solver, results below the Schroeder frequency
  of a room (about 2000 √(T/V) Hz) are a statistical average, not the response of one
  point.
- **Source directivity** was not part of the cross-check (both programs ran
  omnidirectional sources). The tests check that every pattern keeps the sound power,
  that SOFA and CSV data come out in the source's frame, and that the direct sound,
  early paths, late rays and particles all follow the same weights.
- **Sound through walls** is checked against diffuse-field theory, not against
  measurements. A wall is one sound reduction index per band for sound from all
  directions: sound passes straight on in its direction, with no flanking paths and no
  dependence on the angle it arrives at, and the wave solver does not model such walls:
  with `simulation.wave.enabled`, a model with one stops with an error (openings are
  fine).
- **Outdoor propagation** is checked against geometry, closed forms, the exact solution
  for the reflection off a ground and one set of measurements, a wind turbine up to 845
  m away in light winds. A terrain is checked against the exact solution for a wedge,
  the flat ground, the image source in a slope and the thin screen's closed forms, not
  against measurements over hills (see
  [Terrain](#terrain-against-closed-forms-and-the-exact-wedge)), and a rounded crest,
  which has no exact solution, is an engineering model; wind and temperature gradients
  cannot be combined with a terrain yet (see
  [Outdoor scenes](outdoor.md#what-is-not-modelled-yet)), and strong ones, where they
  matter most, are compared with a fast field program on the 1995 benchmark cases (see
  [Outdoor sound in wind](#outdoor-sound-in-wind-against-a-fast-field-program)), not
  with measurements; the benchmark's gradients that go on to a kilometre up bend sound
  through angles the engine's default, narrow-angle parabolic equation does not carry,
  so beyond 1 to 2 km its pure tones stray there, though the bands it reports hold to
  2 km, and the wide-angle one carries them (see
  [The wide-angle parabolic equation](#the-wide-angle-parabolic-equation)). Weather that
  changes over the scene is checked against a parabolic equation marched through the
  zones, not against measurements, and a zone's border is a step (see
  [Weather zones](#weather-zones-against-a-crank-nicholson-parabolic-equation)).
  Levels over a long term are checked against the formulas they come from and against
  ISO 9613-2's meteorological correction, not against long-term measurements, and are
  only as good as the weather classes and shares given to them (see
  [Levels over a long term](#levels-over-a-long-term)).
  Moving sources are checked against the exact field of a quasi-static source over a
  rigid ground, not against measured pass-bys, and leave out the convective
  amplification of a moving source (see
  [Moving sources](#moving-sources-against-the-exact-field)).
  A helicopter's tones are checked against theory, its broadband noise against Pegg's
  worked example, and the whole against one certified level, which it is 3.4 dB over
  (see [Helicopter rotors](#helicopter-rotors-against-theory-and-a-certified-level));
  not against measured spectra or directivity.
  Vegetation is checked against a particle simulation of the same model, not against
  measurements through trees; its foliage takes ISO 9613-2's rates, which the
  standard gives for depths of 20 to 200 m, or leaves as thin rigid discs, whose
  cross-section between its two closed-form limits is bridged, not exact, and which
  are compared with the standard's table, not with measurements of leaves.
  Turbulence is checked against a parabolic equation through random eddies in two
  dimensions, with the source and the listener near the ground, not against
  measurements; how much the sound flickers over time is not checked against either.
  The wave solver in wind and temperature is checked against the parabolic equation
  over open ground up to 200 m, not against measurements, nor round buildings; its
  moving air against the exact field in a uniform wind and a façade's echo against
  ray theory, with the wind the weather's profile everywhere, not the flow round a
  building (see [Moving air](#moving-air-in-the-wave-solver)).
  The level relative to free field and a barrier's insertion loss are checked against
  exact solutions, not against measurements; the path reflected both before and after
  a barrier is checked over a rigid ground only, and the sound that goes over a
  barrier, down to the ground and over it again is left out (see
  [Outdoor results](#outdoor-results-against-exact-solutions)).
- **Spatial outputs** are checked against closed forms and energy balances, not by
  listening tests.
- **Speed on a hardware GPU.** CI checks the wave solver's and the late stage's GPU
  paths on Mesa's software Vulkan driver, which shows that they agree with the CPU, not
  how fast they are; `validation/wave_gpu.py`, `validation/late_gpu.py` and
  `validation/outdoor_gpu.py` time them on your own GPU.

## Reproducing

```sh
python -m pip install pyroomacoustics matplotlib
python validation/pyroomacoustics_crosscheck.py --json results.json --plots docs/validation
python validation/convolver_benchmark.py
python validation/wave_solver.py --plot docs/images/wave.png
python validation/edge_diffraction.py --json diffraction.json
python validation/edge_fade.py --json edge_fade.json
python validation/walks.py --json walks.json   # --walks gallery archway street
python validation/ground_reflection.py --wave --parabolic --extended --json ground.json
python validation/outdoor_propagation.py --json outdoor.json --plots docs/validation
python validation/outdoor_propagation.py --turbulence moderate
python validation/turbulence.py --json turbulence.json
python validation/turbulence.py --profile --json turbulence_profile.json
python validation/vegetation.py --json vegetation.json
python validation/leaves.py --json leaves.json
python validation/wave_outdoor.py --json wave_outdoor.json
python validation/wave_wind.py --json wave_wind.json   # --skip-profiles: no parabolic equation
python validation/insertion_loss.py --json insertion_loss.json
python validation/terrain.py --json terrain.json
python validation/ffp.py --json ffp.json
python validation/benchmark_1995.py --json benchmark.json   # --far: whole scenes to 10 km
python validation/weather_zones.py --json weather_zones.json   # --only pe or mean
python validation/long_term.py --json long_term.json
python validation/doppler.py --json doppler.json
python validation/rotorcraft.py --json rotorcraft.json
python validation/city_speed.py --json city_speed.json   # --sizes 10 20 30 40 60
python validation/wave_gpu.py   # needs the gpu extra and a GPU driver
python validation/late_gpu.py   # the late stage on each device, and its parameters
python validation/outdoor_gpu.py   # the outdoor examples on each device: levels and centre times
python validation/rays_gpu.py   # the ray searches alone, and whether both find the same faces
python validation/bras_validation.py --json bras.json --plots docs/validation
python validation/bras_validation.py --materials initial --plots docs/validation
python validation/bras_validation.py --bands third_octave --json bras_third.json
python validation/bras_validation.py --materials initial --calibrate LS1 --plots docs/validation
```

The cross-check takes about 10 minutes on 4 cores, the benchmark about 2 minutes, the
wave solver's checks about 3.5 minutes (both stencils), the ground's about 7.5 (1.5 of
them the wave solver's, 2 the parabolic equation's), the outdoor measurements' about 2.5
(3 with turbulence), the turbulence's about 7, the vegetation's about 6, the wave
solver's outdoors about 18, the outdoor results' about 3, the terrain's about 4.5, the
leaves' about 2, the fast field program's about 4, the benchmark's about 55 (much
longer with `--far`), the weather zones' about 9 (2 the parabolic equation's, 7 the
mean profile's), the long term's about 0.5, the moving sources' about 3, the fade at edges' about 80 (25 of them the barriers'), the
walks' about 14, the cities' about 1 (2 with `--sizes 10 20 30 40 60`) and each BRAS
run of the seminar room about 4 minutes (8.5 calibrated, 9 in third-octave bands) and of
the hall about 1.5 hours (5.2 calibrated, 3.6 in third-octave bands). The cross-check's numbers come from 4.0.1 with seed 1.
pyroomacoustics draws its rays afresh on every run, so the script averages five runs of it
(`--pra-runs`); the means still move by a few percent from one call to the next, and the
differences in the tables with them. Other seeds move this package's numbers by a few
percent.
