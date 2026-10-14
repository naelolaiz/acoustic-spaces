# Validation

This report sums up how far the results of acoustic_spaces 1.0 can be trusted: what the
test suite checks against theory on every change, how the simulation compares with
another program, [pyroomacoustics](https://github.com/LCAV/pyroomacoustics), and how
much of a block's time the real-time convolver needs. It ends with what has not been
validated yet.

## Checks against theory, on every change

The test suite runs on Linux, macOS and Windows for every pull request and checks:

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
  same absorption on every surface and scattering 0.3 and 1: G within 0.5 dB, C80
  within 2.5 dB (higher from 1 kHz up, where the rays decay up to 7 % slower than
  Eyring);
- the late tail of the example box (absorbing ceiling, smooth walls) from 10k rays
  with splitting against 1 million rays: within 4 dB on average from 0.5 to 2.5 s,
  and splitting or thinning out every ray keeps the energy of plain ray tracing;
- every ISO 3382-1 parameter on synthetic exponential decays, and the air
  absorption against the ISO 9613-1 table;
- spatial outputs: SN3D spherical harmonics against closed forms and orthogonality,
  VBAP gains of unit power that point at the panned direction, the direct sound's
  direction in first-order Ambisonics, the mono energy kept by panned layouts (direct
  sound within 1 %, late part within 5 %), a binaural render through flat HRTFs equal
  to the mono response, and the spherical head's interaural delay (0.6 ms at the side);
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
  off one of the corner's walls within 0.05 dB;
- the wave solver: the modes of a rigid box (within 0.6 Hz), the reflection off the end
  of a duct for four admittances (within 0.005, and below 0.05 for a matched end), the
  free-field level of the 63 Hz band (within 1 dB of the rays), the direction of the
  direct sound, open models and thin screens (stable, and the sound leaves), and runs
  shared by bands with the same absorption;
- sweep measurements of synthetic rooms through a distorting loudspeaker, with
  noise and interface latency: from 250 Hz up, the room's response comes back with
  errors 30 dB below it and T30 within 3 %; with a noise floor 55 dB down, T30 stays
  within 5 % of the noise-free value.

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
Both sets of impulse responses were analysed by the same code
(`acoustic_spaces.parameters.from_impulse_response`), and EDT, T30 and C80 were averaged
over the four listeners, in the octave bands from 125 Hz to 4 kHz.

### The two programs model scattering differently

Reading the ray tracer of pyroomacoustics 0.10.1 (`libroom_src/room.cpp`) shows two
differences that matter here:

- **Where a scattering reflection sends the ray.** Here a ray reflects specularly or,
  with the scattering coefficient *s* as probability, in a random Lambert direction, so a
  share *s* of the energy leaves diffusely: the definition of the coefficient in
  ISO 17497-1. pyroomacoustics sends every ray in the direction of the sum of the
  specular direction and a random Lambert direction, weighted by *s* (its mean over the
  bands): vector-based scattering, as in ODEON. Each reflection then nudges every ray off its specular path a
  little, rather than sending a few rays far off it.
- **How the listener receives scattered energy.** Both programs send the scattered share
  of every reflection straight to each listener that can see the reflection point
  ("diffuse rain"). Here only the specular share is counted again when a ray crosses
  the listener's detector sphere. pyroomacoustics keeps the whole energy in the ray and
  counts all of it at the sphere, so the scattered share reaches the listener twice,
  which raises the reverberant level, lengthens EDT and lowers C80.

To tell which differences come from the scattering model and which from the tracers
themselves, the cross-check script also traces the total sound energy in each room
with either model, without any listener, and fits T30 to the same stretch of its decay
as the listeners' T30 ("room energy" rows below).

### Results

The figures show EDT, T30 and C80 per band for both programs, with the decay of the
room's total energy under each scattering model and, where a diffuse field makes them
apply, Eyring's reverberation time and Barron's revised theory.

**Specular box.** The programs agree: T30 within 4 %, EDT within 7 % and C80 within
1 dB in every band (0.1 to 0.3 dB from 250 Hz up). Rays that travel parallel to the floor
never meet the absorbing ceiling, so the decay is many times longer than Eyring's
estimate in both.

| | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---:|---:|---:|---:|---:|---:|
| EDT (s), acoustic_spaces | 7.29 | 6.32 | 3.55 | 3.01 | 2.33 | 1.33 |
| EDT (s), pyroomacoustics | 7.82 | 6.28 | 3.47 | 2.96 | 2.17 | 1.27 |
| EDT difference | +7 % | +0 % | -2 % | -2 % | -7 % | -4 % |
| T30 (s), acoustic_spaces | 15.05 | 13.98 | 8.32 | 6.31 | 4.98 | 3.05 |
| T30 (s), pyroomacoustics | 15.30 | 13.81 | 8.25 | 6.31 | 5.06 | 3.16 |
| T30 difference | +2 % | -1 % | -1 % | +0 % | +2 % | +4 % |
| C80 (dB), acoustic_spaces | -2.7 | -1.4 | 0.0 | 1.7 | 3.1 | 5.0 |
| C80 (dB), pyroomacoustics | -1.6 | -1.4 | 0.3 | 1.9 | 3.2 | 5.1 |
| C80 difference | +1.0 dB | +0.1 dB | +0.3 dB | +0.1 dB | +0.1 dB | +0.1 dB |
| T30 (s) of the room's energy, random scattering | 15.59 | 14.04 | 7.31 | 6.30 | 4.97 | 2.86 |
| T30 (s) of the room's energy, vector-based scattering | 15.60 | 14.03 | 7.27 | 6.30 | 4.97 | 2.87 |
| Eyring (s) | 0.83 | 0.69 | 0.52 | 0.37 | 0.30 | 0.25 |

![Specular box](validation/crosscheck_specular.png)

**Diffuse box.** T30 agrees within 6 % (within 4 % except at 250 Hz); both programs
stay within 6 % of the room's energy decay and within 11 % of Eyring's estimate. From 1 kHz up, pyroomacoustics gives 13 to 21 % longer
EDT and 1.6 to 2.6 dB lower C80 than this package, as its double-counted scattered energy
predicts. Barron's revised theory sides with this package: computed from the simulated
energies, as `simulate` prints them, C80 is within 0.7 dB of the theory in every band.
From the impulse responses it is up to 1.9 dB lower at 250 and 500 Hz, in both programs
alike, because the direct sound and the first floor and ceiling reflections interfere
within those octave bands.

| | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---:|---:|---:|---:|---:|---:|
| EDT (s), acoustic_spaces | 0.91 | 0.70 | 0.61 | 0.38 | 0.31 | 0.26 |
| EDT (s), pyroomacoustics | 0.83 | 0.79 | 0.62 | 0.43 | 0.37 | 0.32 |
| EDT difference | -9 % | +13 % | +2 % | +13 % | +18 % | +21 % |
| T30 (s), acoustic_spaces | 0.75 | 0.68 | 0.53 | 0.40 | 0.30 | 0.26 |
| T30 (s), pyroomacoustics | 0.74 | 0.63 | 0.55 | 0.40 | 0.32 | 0.27 |
| T30 difference | -1 % | -6 % | +3 % | +1 % | +4 % | +3 % |
| C80 (dB), acoustic_spaces | 4.9 | 5.6 | 7.1 | 12.8 | 15.5 | 18.9 |
| C80 (dB), pyroomacoustics | 4.5 | 4.5 | 7.0 | 10.8 | 13.9 | 16.3 |
| C80 difference | -0.4 dB | -1.1 dB | -0.1 dB | -2.0 dB | -1.6 dB | -2.6 dB |
| C80 (dB), Barron's revised theory | 5.2 | 6.7 | 9.0 | 13.0 | 16.3 | 19.6 |
| T30 (s) of the room's energy, random scattering | 0.79 | 0.66 | 0.53 | 0.38 | 0.31 | 0.26 |
| T30 (s) of the room's energy, vector-based scattering | 0.78 | 0.66 | 0.52 | 0.37 | 0.30 | 0.26 |
| Eyring (s) | 0.83 | 0.69 | 0.52 | 0.37 | 0.30 | 0.25 |

![Diffuse box](validation/crosscheck_diffuse.png)

**Example box.** With smooth walls (scattering 0.05) the programs disagree by 19 to 42 %
on T30: rays parallel to the floor survive much longer when only one reflection in
twenty sends them elsewhere than when every reflection tilts them a little. Each program
matches the room-energy decay of its own scattering model within 5 %, so the difference
is the choice of model, not an error in either tracer. EDT and C80 follow the same split
below 1 kHz.

| | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---:|---:|---:|---:|---:|---:|
| EDT (s), acoustic_spaces | 1.75 | 1.69 | 1.41 | 1.16 | 1.04 | 0.81 |
| EDT (s), pyroomacoustics | 1.40 | 1.30 | 1.20 | 1.23 | 1.14 | 0.88 |
| EDT difference | -20 % | -23 % | -15 % | +6 % | +10 % | +9 % |
| T30 (s), acoustic_spaces | 2.96 | 2.80 | 2.38 | 2.20 | 2.02 | 1.59 |
| T30 (s), pyroomacoustics | 1.70 | 1.69 | 1.56 | 1.52 | 1.46 | 1.28 |
| T30 difference | -42 % | -40 % | -34 % | -31 % | -28 % | -19 % |
| C80 (dB), acoustic_spaces | 1.2 | 1.4 | 2.3 | 4.8 | 5.5 | 7.1 |
| C80 (dB), pyroomacoustics | 2.3 | 3.4 | 3.9 | 4.9 | 5.0 | 6.4 |
| C80 difference | +1.1 dB | +2.0 dB | +1.6 dB | +0.1 dB | -0.5 dB | -0.7 dB |
| T30 (s) of the room's energy, random scattering | 2.82 | 2.73 | 2.34 | 2.21 | 2.04 | 1.57 |
| T30 (s) of the room's energy, vector-based scattering | 1.67 | 1.66 | 1.54 | 1.50 | 1.43 | 1.22 |
| Eyring (s) | 0.83 | 0.69 | 0.52 | 0.37 | 0.30 | 0.25 |

![Example box](validation/crosscheck_example.png)

**Lecture hall.** The same picture, smaller because the audience floor scatters
strongly: T30 differs by 19 to 23 % at 125 and 250 Hz and by less than 10 % from 500 Hz
up. pyroomacoustics matches its model's room-energy decay within 5 %, this package
within 10 %, probably because the listeners sit 1.2 m above the absorbing audience,
where the sound field is less uniform than in the room as a whole.

| | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
|---|---:|---:|---:|---:|---:|---:|
| EDT (s), acoustic_spaces | 2.31 | 2.13 | 1.58 | 1.38 | 1.34 | 1.14 |
| EDT (s), pyroomacoustics | 1.83 | 1.70 | 1.44 | 1.44 | 1.45 | 1.17 |
| EDT difference | -21 % | -20 % | -9 % | +4 % | +8 % | +3 % |
| T30 (s), acoustic_spaces | 2.52 | 2.45 | 2.16 | 2.09 | 1.97 | 1.60 |
| T30 (s), pyroomacoustics | 1.94 | 1.99 | 1.96 | 1.97 | 1.94 | 1.66 |
| T30 difference | -23 % | -19 % | -9 % | -6 % | -1 % | +3 % |
| C80 (dB), acoustic_spaces | -0.5 | -0.8 | 1.1 | 2.8 | 3.2 | 4.0 |
| C80 (dB), pyroomacoustics | -0.4 | -0.3 | 1.9 | 2.2 | 3.1 | 3.6 |
| C80 difference | +0.1 dB | +0.5 dB | +0.8 dB | -0.6 dB | -0.1 dB | -0.3 dB |
| T30 (s) of the room's energy, random scattering | 2.52 | 2.48 | 1.97 | 1.94 | 1.86 | 1.54 |
| T30 (s) of the room's energy, vector-based scattering | 1.97 | 1.98 | 1.91 | 1.95 | 1.90 | 1.58 |
| Eyring (s) | 1.99 | 1.89 | 1.13 | 0.99 | 0.88 | 0.86 |

![Lecture hall](validation/crosscheck_hall.png)

### Conclusions

- With purely specular or fully diffusing surfaces, where both programs model the same
  physics, T30 agrees within 6 % and, in the specular room, EDT within 7 % and C80
  within 1 dB. That is close to the 5 % and 1 dB the plan aimed for; T30 from 100,000
  rays also varies by a few percent from one random seed to the next.
- With partly scattering surfaces the programs differ by up to 42 % in T30 because they
  model scattering differently: each matches the room-energy decay of its own model.
  Measurements, not another program, have to decide which model predicts real rooms
  better; vector-based scattering could be added here as an option for comparisons with
  ODEON.
- Where the theory is clear (the diffuse box), this package agrees with Eyring and with
  Barron's revised theory; pyroomacoustics counts the scattered energy twice.

## Wave-based solver

`validation/wave_solver.py` checks the wave solver against theory: closed forms where
there are any, and for a whole room the sum of its modes.

**Modes and their decay.** A 7 x 5 x 3 m box with every wall of specific admittance
0.005 (10 cm cells) rings for 8 s. For each mode that stands apart from the others, the
frequency of its peak and the decay rate of its envelope match the rigid box's
frequency and the decay rate of perturbation theory, δ = cβ/2 Σ (4/L along each axis
the mode varies, 2/L along the others), within 0.4 %:

| mode | theory (Hz) | wave solver (Hz) | decay, theory (1/s) | wave solver (1/s) |
|---|---:|---:|---:|---:|
| (1, 0, 0) | 24.50 | 24.48 | 1.405 | 1.404 |
| (0, 1, 0) | 34.30 | 34.31 | 1.503 | 1.501 |
| (1, 1, 0) | 42.15 | 42.15 | 1.748 | 1.745 |
| (2, 0, 0) | 49.00 | 49.00 | 1.405 | 1.403 |
| (0, 0, 1) | 57.17 | 57.11 | 1.731 | 1.727 |
| (0, 2, 1) | 89.30 | 89.29 | 2.074 | 2.067 |

Axial modes lose energy at walls they meet head-on and at walls they graze, so this
checks absorption at all angles.

**Walls.** A plane wave down a duct (10 cm cells) reflects off its end as
(1 − β)/(1 + β) predicts at normal incidence, and grazing its absorbing walls it decays
as exp(−βP/(2A)) per metre (perimeter P, cross-section A), whatever the cell size:

| end admittance β | 0 | 0.1 | 0.25 | 0.5 | 1 |
|---|---:|---:|---:|---:|---:|
| reflection, theory | 1.000 | 0.818 | 0.600 | 0.333 | 0 |
| wave solver (60 to 150 Hz) | 0.999 | 0.818 | 0.601 | 0.336 | 0.044 |

| walls β = 0.05, amplitude after 10 m | theory | wave solver, 10 cm cells | 5 cm cells |
|---|---:|---:|---:|
| 0.6 x 0.6 m duct | 0.1889 | 0.1900 | 0.1906 |

**Free field.** In a 30 x 24 x 24 m room with 95 % absorption, the direct sound's
energy in the 63 and 125 Hz bands at 1 to 8 m is within 0.35 dB of the free-field level
the rays give (−0.06 to −0.13 dB at 63 Hz, −0.16 to −0.33 dB at 125 Hz).

**A whole room.** A 7 x 5 x 3 m box with 15 % absorption on every surface, simulated
with the default settings (the 63 and 125 Hz bands from the wave solver, 20 cm cells),
against the sum of the box's modes up to 1.5 kHz, each with its decay rate, all
analysed alike (ISO 3382-1, octave filters):

| model | G 63 Hz (dB) | G 125 Hz | EDT 63 Hz (s) | EDT 125 Hz | T20 63 Hz (s) | T20 125 Hz |
|---|---:|---:|---:|---:|---:|---:|
| sum of the modes | 20.3 | 23.5 | 1.02 | 0.96 | 0.94 | 0.90 |
| wave solver | 21.1 | 22.7 | 1.04 | 0.92 | 0.94 | 0.89 |
| wave solver, 16 cells per wavelength | 20.8 | 23.6 | 1.01 | 0.81 | 0.95 | 0.94 |
| rays only | 25.7 | 21.9 | 0.67 | 0.77 | 0.92 | 0.83 |

![Transfer function of the box from the sum of its modes, the wave solver and rays](images/wave.png)

The wave solver follows the modes peak for peak from 22 Hz up. What is left comes
mostly from where the grid puts the walls: within half a cell of the model's, here
6.93 x 4.95 x 2.97 m with 20 cm cells, which moves each mode by up to 1 %. In a room
this sparse in modes, that moves G by up to 1 dB and EDT by up to 16 %; more cells per
wavelength move the walls but do not make every figure converge. Rays give the average
over the room instead: 5.4 dB more at this seat in the 63 Hz band, where the listener
sits near the nodes of most modes.

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

## Not validated yet

- **Measurements.** No simulated room has been compared with measurements of the real
  one yet. The BRAS database (Aspöck et al., TU Berlin) has measured rooms with acoustic
  material data, but its models are SketchUp files that need exporting to OBJ first.
- **Wave effects.** Above the wave solver's bands the simulation is geometric. Edge
  diffraction is modelled (UTD, for the direct sound and first-order reflections) and
  room modes by the wave solver, both checked against theory but not against
  measurements in rooms. Without the wave solver, results below the Schroeder frequency
  of a room (about 2000 √(T/V) Hz) are a statistical average, not the response of one
  point.
- **Source directivity** was not part of the cross-check (both programs ran
  omnidirectional sources). The tests check that every pattern keeps the sound power,
  that SOFA and CSV data come out in the source's frame, and that the direct sound,
  early paths, late rays and particles all follow the same weights.
- **Spatial outputs** are checked against closed forms and energy balances, not by
  listening tests.

## Reproducing

```sh
python -m pip install pyroomacoustics matplotlib
python validation/pyroomacoustics_crosscheck.py --json results.json --plots docs/validation
python validation/convolver_benchmark.py
python validation/wave_solver.py --plot docs/images/wave.png
```

The cross-check takes about 6 minutes on 4 cores, the benchmark about 2 minutes and the
wave solver's checks about 30 seconds. The
numbers above come from acoustic_spaces 1.0.0 with seed 1; other seeds move them by a few
percent.
