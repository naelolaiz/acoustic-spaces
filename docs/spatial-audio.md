# Spatial audio and listening

## Spatial outputs

The scene's `outputs` list says which formats `simulate` writes (mono by default):

```yaml
outputs:
  - mono
  - binaural                              # spherical head model
  - {format: binaural, hrtf: kemar.sofa}  # measured HRTFs
  - {format: ambisonics, order: 3}        # AmbiX: ACN order, SN3D, orders 0 to 7
  - {format: loudspeakers, layout: "5.1"} # stereo, 5.0, 7.0, 5.0.4, 7.0.4, and .1 with LFE
  - format: loudspeakers                  # or any layout, in channel order
    speakers: [{name: L, azimuth: 30}, {name: R, azimuth: -30}, {name: Top, azimuth: 0, elevation: 60}]
```

Directions are those of the listener's head as its `orientation` turns it: azimuth
counts to the left, elevation up, as in SOFA and AmbiX. Every early reflection keeps
its exact arrival direction. For the late part the stochastic tracer sorts arriving
energy into 64 directions (`simulation.late.directions`) every 10 ms; each impulse of
the synthesized tail then comes from one of them, drawn in proportion to that energy.
The statistical tail arrives evenly from all around.

- **Ambisonics** channels are the real spherical harmonics with SN3D normalisation, so
  W is the mono response exactly.
- **Loudspeakers** get vector base amplitude panning (VBAP, unit power). Layouts with
  height but nothing below the listener get an imaginary speaker straight down whose
  share is dropped. A gap of more than 90 degrees between the speakers on the horizon,
  as behind the listener in 5.x.4, gets imaginary speakers on the horizon that the two
  speakers either side play, so arrivals there pan as in 5.0. Four or more speakers on
  one plane, as the height ring of 5.x.4 and 7.x.4 and the rear corners of 7.x.4, get
  one at their middle that they all play, so a symmetric layout pans symmetrically and
  straight up plays from the whole ring. Named layouts carry their WAV channel mask;
  the LFE channel of the `.1` layouts stays silent, leaving bass management to the
  playback system.
- **Binaural** gives every reflection and each late direction an HRTF interpolated
  between the measured directions: the measured directions are triangulated as for
  VBAP, and the responses at the corners of the triangle a direction falls in are
  aligned at their onsets, mixed with its barycentric weights and delayed by the same
  mix of their onsets, so the interaural time difference follows the direction smoothly
  even with a coarse set. A response's onset is the lag at which it best matches its
  minimum-phase version, to a fraction of a sample, so responses delayed by fractions
  of a sample still add in phase and the mix keeps their spectrum (within 1 dB).
  Without a SOFA file it uses Brown and Duda's spherical head model: interaural time
  and level differences, but no pinna cues. `read_sofa` reads SimpleFreeFieldHRIR
  files, which most HRTF databases (for example those listed by
  [sofaconventions.org](https://www.sofaconventions.org)) provide; it needs h5py.

Levels match the mono response: an arrival from straight ahead has the same height in
W, in the speaker it is panned to, and, below the head's influence, at both ears.

### Microphones

A listener can be a microphone: its `pattern` is what its mono output picks up through,
aimed where the listener faces.

```yaml
listeners:
  - name: vocal_mic
    position: [0.0, 0.3, 1.6]
    orientation: {yaw: 180, pitch: -10}
    pattern: supercardioid        # omni (the default), subcardioid, cardioid,
                                  # supercardioid, hypercardioid or figure8
  - name: main_pair
    position: [6.0, 0.0, 3.0]
    orientation: {yaw: 180, pitch: -15}
    pattern: {pair: xy, angle_deg: 90, capsule: cardioid}
  - name: blumlein
    position: [8.0, 0.0, 3.0]
    orientation: {yaw: 180}
    pattern: {pair: blumlein}     # two figure8s aimed 45 degrees to each side
  - name: ms
    position: [10.0, 0.0, 3.0]
    orientation: {yaw: 180}
    pattern: {pair: ms, mid: cardioid, side_db: 0}
```

- A first-order pattern picks up a + (1 - a) cos θ of sound arriving θ off its aim,
  with the `a` the sources' patterns take (0.7, 0.5, 0.37, 0.25 and 0 from subcardioid
  to figure8): 1 straight ahead, the same at every frequency, as an ideal capsule.
- A coincident pair writes two channels, left and right, in the mono output's file:
  `xy`, two capsules of one pattern (cardioids by default) aimed `angle_deg` apart (90
  by default), the left one to the left of the front; `blumlein`, two figure8s aimed 45
  degrees to each side; `ms`, a `mid` capsule aimed forward and a figure8 aimed left,
  decoded to M + gS on the left and M - gS on the right, g the side's gain `side_db` (0
  dB by default; more is wider). The report's mono output names the microphone
  (`"microphone": "xy"`, say) and its channels.
- Each early path is picked up from the direction it arrives from, and each impulse of
  the late part from the one it is drawn from, so a pattern's response is the first
  order of the field the simulation gives (to the 64 directions the tracer sorts the
  late energy into), and the wave solver's bands are their first-order field too. A
  pair's channels hold the same late impulses, as two capsules in one place hear the
  same reverberation.
- The pattern changes what the mono output holds, in `simulate`'s files, `auralize`,
  walkthroughs, movements and the editor's listening; not the parameters
  (`parameters.json` and the editor's results stay those of an omnidirectional
  listener, as ISO 3382 measures them), the echo check, or the binaural, Ambisonics and
  loudspeaker outputs. `validate` warns about a pattern in a scene whose outputs leave
  mono out.

Spaced pairs (AB, ORTF, a Decca tree) are a listener for each capsule, each with its
pattern and its orientation. Their direct sound and early reflections arrive with the
delays between the capsules' places, but each listener's late part is drawn from noise
of its own, so their reverberation does not correlate at any frequency, where real
capsules d apart correlate below about c / 2d (1 kHz for 17 cm): the bass of such a
pair sounds wider than it would.

### HRTF sets to download

A measured head places sounds better than the spherical head model, front from back
above all. `acoustic-spaces hrtf` lists the free sets this program knows, with their
licences, and downloads the one named into the user's data folder
(`--folder` puts it elsewhere, `--url` fetches any other SOFA file):

```sh
acoustic-spaces hrtf                # the sets, their licences and sizes
acoustic-spaces hrtf ku100          # download one; it prints the file
acoustic-spaces hrtf viking --member subj_K.sofa   # one head of a set of many
acoustic-spaces simulate scene.yaml --out results --hrtf ~/.local/share/acoustic-spaces/hrtf/ku100.sofa
```

| Set | Ears | Licence | Download | What it holds |
| --- | --- | --- | --- | --- |
| `ku100` | a Neumann KU100 dummy head | CC BY 3.0 | 3.4 MB | 2702 directions, 48 kHz, 128 taps |
| `kemar` | a KEMAR dummy head (SADIE II) | Apache 2.0 | 38 MB | 8802 directions, 48 kHz, 256 taps |
| `viking` | one of 20 people, and a dummy head | CC BY 4.0 | 58.4 MB | 1513 directions, 48 kHz, 128 taps |

The sets are other people's measurements: `ku100` is Benjamin Bernschütz's compilation
at TH Köln (<https://doi.org/10.5281/zenodo.3928297>), `kemar` the University of York's
SADIE II database (<https://doi.org/10.5281/zenodo.12092466>) and `viking` the Viking
dataset v2 of Spagnol, Miccini and Unnthorsson at Reykjavík University
(<https://doi.org/10.5281/zenodo.4160401>), which holds one file per head. Each licence
asks for its credit in a work that uses it; the command and the editor both print it
before downloading. A dummy head measures the same way every time; a person's own ears
place sounds better for that person and worse for everybody else.

In the editor, **Download HRTFs…** on the **Outputs** page (and in the **Run demo…**
window beside the HRTF field) shows the same sets with their licences, downloads the
one chosen in the background and puts it in the scene's binaural output. Any other
SOFA file goes in by its address in the same window, or with **HRTF file…** from the
disk.


## Listening: convolution and real-time playback

Run a dry (anechoic) recording through a simulated or measured response:

```sh
acoustic-spaces convolve speech.wav results/talker__seat-a.binaural.wav --out wet.wav
acoustic-spaces auralize results talker=speech.wav piano=piano.wav --format 5.0 --out mix.wav
acoustic-spaces play results/talker__seat-a.binaural.wav speech.wav   # real time
acoustic-spaces play results/talker__seat-a.binaural.wav --live      # microphone in
```

`convolve` takes any response WAV, with any number of channels; `auralize` reads a
`simulate` folder and mixes every source's own signal as heard at one `--listener`, in
one of the formats it wrote (`mono`, `binaural`, `ambix3`, `5.0`, ...), keeping the
levels the simulation gives each source. The responses carry each source's feed
(`gain_db`, `eq_db`, `delay_ms`) but are per unit of sound power, so each signal is
scaled by its source's `level_db` against the loudest one's in the mix (from the sound
powers `parameters.json` lists): a front fill 18 dB below the main loudspeakers plays
18 dB softer, each signal taken at the same level. `--equal-power` mixes them as if all
had the loudest one's power; results written by an older version, which lack the
powers, are mixed that way, with a note. `walkthrough` does the same. A binaural
output through a SOFA file is written as `binaural-<file name>`
(`talker__seat-a.binaural-kemar.wav` for `kemar.sofa`), so that it and the spherical head's can be written together; `--format
binaural` finds it when it is the only binaural one. Dry signals are mixed down to
mono, as sources are points, and resampled to the response's rate. The results are
32-bit float WAV files; `--normalize` scales them to a -1 dBFS peak.

`play` convolves in real time on the sound card (it needs the `measure` extra). The
compiled convolver adds no latency beyond one `--block` of frames (256 by default, 5.3
ms at 48 kHz), and each block is written into buffers made before playback starts. It
cuts the response into partitions that grow along it, from the block size up to 8192
frames, and spreads the work of the long ones evenly over the blocks in between
(non-uniform partitioning), so long and multichannel responses stay cheap: a 3 s,
16-channel third-order Ambisonics response in blocks of 128 frames takes about 0.3 ms of
the 2.7 ms each block lasts, on one current x86-64 core, where uniform partitions took
3.6 ms. By default the response is scaled to unit energy so the result plays about as
loud as the dry signal; `--absolute` keeps the simulated level.

From Python:

```python
from acoustic_spaces.render import Convolver, auralize, convolve, play, resample
from acoustic_spaces.spatial import Binaural

mix = auralize(result, {"talker": (speech, 44_100)}, "seat-a", Binaural())  # (samples, 2)
wet = convolve(speech_48k, binaural)  # offline, float64
convolver = Convolver(binaural, block=256)  # block by block, float32
out = convolver.process(next_block)  # (256, 2)
convolver.process(next_block, out=out)  # into an array of your own, as a callback would
play(binaural, 48_000, speech_48k)  # through the sound card
```

### Dry recordings to try

These anechoic recordings are free to download, and each one tests something different:

- **BRAS anechoic audio** ([BRAS](https://doi.org/10.14279/depositonce-6726.3), folder
  `4 Additional data/Audio/AnechoicAudio`; CC BY-SA 4.0; WAV, 32-bit float, 44.1 kHz):
  noise pulses and a string quartet with one instrument per channel. Next to them,
  `Audio/Auralization` holds the authors' renderings through the measured BRIRs of each
  room (quartet channels 1 to 4 from LS3 to LS6, the noise pulses from LS7, all heard at
  MP6), so a simulation of the same room, sources and seat can be heard side by side
  with the real one. The noise pulses make the decay and its colour easy to hear.
- **Beethoven, Symphony No. 8** ([TU Berlin](https://doi.org/10.14279/depositonce-6729.2);
  Böhm, Ackermann and Weinzierl; CC BY 4.0; WAV, 32-bit float, 48 kHz): movements
  1, 2 and 4 of a full orchestra, 61 mono tracks per movement (zips of 590 to 740 MB),
  for `auralize` with one source per instrument on a stage, as in the
  [`orchestra` example](demos.md#an-orchestra-in-the-concert-hall).
- **Speech**: Cicero's [3rd Catiline Oration](https://doi.org/10.14279/depositonce-8536)
  in Italian, Latin and German, and Demosthenes' [1st Olynthic
  Oration](https://doi.org/10.14279/depositonce-8537) in German (TU Berlin; CC BY-NC-SA
  4.0, non-commercial; mono WAV, 48 kHz), for clarity, intelligibility and echoes in
  lecture rooms and churches.
- **University of Iowa [Musical Instrument
  Samples](https://theremin.music.uiowa.edu/MIS.html)** (free for any use; AIFF, 16-bit
  44.1 kHz, or 24-bit 96 kHz for those recorded since 2011): single notes of orchestral
  instruments, all but the piano recorded in an anechoic chamber. Short attacks
  (xylophone, marimba, pizzicato) make discrete echoes and any click at the convolver's
  partition boundaries easy to hear.
- **Aalto anechoic symphonic recordings**
  ([Aalto University](https://research.cs.aalto.fi/acoustics/virtual-acoustics/research/acoustic-measurement-and-analysis/85-anechoic-recordings.html);
  Pätynen, Pulkki and Lokki, 2008; free for academic research; MP3, PCM on request):
  orchestral excerpts recorded one instrument at a time.

The commands read WAV files only (16, 24 or 32-bit PCM, or float), so convert AIFF, FLAC
or MP3 first, for example with `ffmpeg -i note.aiff -c:a pcm_f32le note.wav`. A dry file
at another rate is resampled to the response's, with a note. A file with several
channels is averaged to mono, so split a multitrack file into one file per source
first; channels that are present but silent also lower the level (the BRAS files have
five channels, of which the noise pulses use one and the quartet four):

```python
from acoustic_spaces.audio import read_wav, write_wav

quartet, rate = read_wav("StringQuartet_4channel.wav")
for n in range(4):
    write_wav(f"quartet_{n + 1}.wav", quartet[:, n], rate)
```

A unit impulse at the response's rate checks the convolution itself: `convolve` gives
the response back to float precision.

### Demos

Three [demos](demos.md) mix dry recordings through example scenes and make a video that
tours their listeners, with the scripts that do it: an orchestra in the concert hall
(the Beethoven recording above), a choir and an organ in the church, and a band with its
sound system and monitors in the stadium.

### Walkthroughs

A listener or a source with a [walk](scenes.md#walks) can be heard as it goes. In the
`gallery` example a visitor walks away from a guide, down the long arm of the L-shaped
gallery and round the corner, while the guide talks (any dry speech recording does as
`speech.wav`; see [dry recordings to try](#dry-recordings-to-try)); the gallery's two
other listeners stand still, and with the visitor's walk turned off (**Moves**, or
`enabled: false`) the visitor stands too, so one scene serves both:

```sh
acoustic-spaces examples --copy . gallery && cd gallery
acoustic-spaces walkthrough scene.yaml guide=speech.wav --out walk.wav --video walk.mp4
```

- Responses are simulated every `--step` along the path (25 cm by default, 101 places
  for the gallery's 25 m). The early sound, which changes quickly from place to place,
  is simulated at every one of them; the late field changes slowly, so the late part,
  by far the costliest to trace, is traced only every `--late-step` (2 m by default,
  14 places), and the places in between mix the late parts of the two on either side
  in proportion to how near they are (with the statistical late method, which costs
  little, every place has its own). Along the gallery, late parts mixed from places
  4 m apart, twice the default, came within 0.3 dB on average of those traced at the
  places themselves, band by band, and within 1.3 dB next to the corner, where the
  field changes fastest (with 10,000 rays per band, whose own noise is part of that).
  The gallery's walk takes about five minutes on 4 cores, six times as long as one
  response.
- The dry signals play through the response of the place the walker is nearest to,
  with a `--crossfade` (a raised cosine, 50 ms by default) half-way between two places,
  so the sound changes smoothly as it walks: the crossfades' weights always add up to
  one, and one response at every place gives exactly the convolution with it. The
  raised cosine changes the gains without the corners of a linear crossfade, whose
  sudden change of slope spreads a tone's energy to higher frequencies, and never
  faster than π/2 times a linear crossfade's mean slope (the tests). Every place's late part is drawn from the same random
  sequence, so the tails that cross fade differ only in their level and decay, not in
  their noise. Once the walk is over, the walker stays where it ends for as long as the
  signals last.
- A walking listener hears every source; a walking source is heard at `--listener`,
  with the other sources where they stand. `--format` takes the formats of `simulate
  --output` (by default the scene's first output, or binaural with `--hrtf`, which is
  for binaural only) and `--video` saves the room, the path and the walker going along
  it, with an arrow and its yaw for the way it faces, as a GIF or an MP4 (both with the
  viz extra) at `--fps`. An MP4 carries the sound when the format has one or two
  channels (mono, stereo, binaural) and then lasts as long as the sound; with more
  channels, play the WAV alongside it.
- A walker faces the way it goes (unless its walk says `face_forward: false`) and
  turns through each corner at 90 degrees per second. The gallery's visitor has the
  guide behind it all down the long arm, and at the corner turns over a second towards
  the short arm, which brings the guide round to its right. The default binaural head
  is a sphere, which tells left from right but hardly front from behind (it has no
  outer ears), so the guide behind sounds much as one ahead would; a measured HRTF
  (`--hrtf`) tells them apart.
- At walking speed the Doppler shift is under half a percent (8 cents), and the
  crossfades leave it out. A car, a train or an aircraft takes `doppler: true` in its
  walk (see [Moving sources and Doppler](#moving-sources-and-doppler)).
- A walk along [keyframes](scenes.md#walks) (timed places and facings, so the walker
  can wait, turn on the spot or go at any pace) is placed every `--step` along each leg
  and every `--turn-step` (20° by default) of a turn on the spot, and at every keyframe.
- Outdoors in wind or a temperature gradient, the places of a walking source share the
  parabolic equation's runs, with the results of a run per place: an aircraft's
  flyover in the `windy` preset runs in about a third of the time (see
  [How the simulation bends sound](outdoor.md#how-the-simulation-bends-sound)).

From Python:

```python
from acoustic_spaces import Room
from acoustic_spaces.spatial import Binaural
from acoustic_spaces.viz import animate_walk
from acoustic_spaces.walkthrough import render_walk, simulate_walk

room = Room.from_yaml("scene.yaml")
walk = simulate_walk(room)  # the only listener or source with a walk
sound = render_walk(walk, {"guide": (speech, 44_100)}, Binaural())  # (samples, 2)
animate_walk(room, walk, "walk.gif")
```

#### Several movers at once

When more than one listener or source has a walk, `walkthrough --all` plays them all
together, on one clock, in one simulation:

```sh
acoustic-spaces walkthrough scene.yaml band=song.wav speaker=talk.wav --all --out heard.wav
```

- Every walk that is `enabled` moves (the default; `enabled: false` leaves a walker
  standing where it starts), and every listener in use hears every source in use, still
  or moving, so the balance between listeners is the scene's. One WAV is written per
  listener (`heard_<listener>.wav`; `--listener NAME` for one of them, as `--out`
  names it).
- The moments simulated are every mover's places: every `--step` along its walk, every
  `--turn-step` of a turn on the spot, and its keyframes or corners. At each moment each
  mover stands as a copy of itself where its walk has it then, one copy serving every
  moment it stands still at (once its walk is over, say). All the copies go into one
  simulation with the still sources and listeners, so a moving listener costs what a
  walk of its own does, and the late part is traced every `--late-step` along each walk,
  the places between mixing theirs, as for one walk. A scene where both a source and a
  listener move simulates every copy of the one against every copy of the other, which
  grows with the square of the moments: keep `--step` coarse there, or let one of them
  stand.
- A pair that moves crossfades from each moment's response to the next's, as a walk
  does; a still pair plays through its one response. `--video` draws every mover along
  its path with an arrow the way it faces, and the still sources and listeners where
  they stand; an MP4 carries the sound when one listener is rendered.
- A walk with Doppler moves alone: `--all` plays it when it is the only walk that is
  on, every listener hearing each early path through its own changing delay as
  `--walker` does, from one simulation, and refuses it beside another mover. `--levels`
  is for one walk (`--walker`). The demo of any scene and the editor's **Render
  movement…** play a walk with Doppler the same way
  ([demos](demos.md#a-demo-of-any-scene)).

From Python: `simulate_movement(room)`, `render_movement(movement, signals, fmt)` (a
dict of each listener's sound), `save_movement(rendered, "heard.wav", rate, fmt)` and
`animate_movement(room, movement, "heard.mp4")`. In the editor, the Movement panel
edits the keyframes, previews every mover along one timeline and renders the movement
in the background (see [Graphical editor](editor.md)).

### Moving sources and Doppler

A source that passes quickly, a car, a train or an aircraft, drops in pitch as it goes
by, and so does a source that a listener passes quickly. With `doppler: true` in its
[walk](scenes.md#walks), the walker's early sound follows it:

```yaml
sources:
  - name: car
    position: [-60.0, 0.0, 0.5]
    power_db: 94
    walk: {to: [[60.0, 0.0, 0.5]], speed_m_s: 25, face_forward: false, doppler: true}
```

```sh
acoustic-spaces walkthrough scene.yaml car=engine.wav --listener kerb --out pass.wav --levels pass.csv
```

- Each early path (the direct sound, each reflection, the sound bent round each edge)
  is followed from place to place and played through a delay that changes smoothly as
  the walker goes: a cubic curve through the path's delay at each place, read through a
  windowed sinc that cuts off below the Nyquist frequency as the pitch rises, so that
  nothing aliases. A walking source is heard at the retarded time, when the sound it
  sent from each place arrives; a walking listener hears what reaches it at each place.
  So each path shifts in pitch as its own length changes: the ground's reflection
  shifts a little differently from the direct sound, and their interference sweeps as
  it does by a road. Each path keeps its filter (the ground's, the air's, a barrier's,
  the ears' in binaural), which changes from one place to the next as before, and in
  turbulence the eddies drift past as in any walkthrough.
- The late part is crossfaded from place to place as without Doppler, its pitch
  unshifted; outdoors it is far below the early sound.
- Against the exact field of a source passing 10 m from a listener over a hard ground,
  the sound is within 45 dB of it at 30 m/s, with places 1 m apart, and its level
  within 0.04 dB at every moment; a train at 80 m/s is within 38 dB and 0.42 dB at
  4 kHz. Crossfading from place to place instead misses the level by 4 to 11 dB at
  times, where the two paths interfere (see
  [Validation](validation.md#moving-sources-against-the-exact-field)). The places
  should be no more than 1 m apart (the default `--step` is 25 cm): 2 m apart the
  sound is within 38 dB of the field, 4 m apart only within 18 dB.
- The source is quasi-static: its sound power and directivity are those of a source
  standing at each place. A moving source's convective amplification, which makes it
  louder ahead of it and quieter behind, is left out: taking its intensity as
  (1 − M cos θ)⁻², one common form whose exponent depends on the source, it would add
  up to 0.8 dB ahead at 30 m/s and 2.0 dB at 70 m/s, and take 0.7 and 1.6 dB behind.
- **The sound sent each way.** From Python, the walker (or, for a walking listener, a
  source it passes) may send an `Emission` (`acoustic_spaces.emission`) instead of one
  signal: the sound it sends in every direction of its own frame, as a rotor sends
  each harmonic of its blades with a pattern and a phase of its own. Each early path
  then carries what is sent the way it leaves the source (its first leg, turning from
  one place's way to the next's) at the time it leaves, and the source's directivity
  is taken out of the path's filter; the late part plays the emission's sound averaged
  over the directions. `Mixture` makes one of signals that do not correlate, each
  with an amplitude per direction; `Periodic` one of a sound that repeats, one period
  of it in each direction, which may turn with a rotor and be sent from off the
  source's place (a helicopter's tones, see [Helicopters](outdoor.md#helicopters));
  `Sum` adds emissions up. The directivity and the sound power still give the band
  levels the simulation and the pass-by report, so give the source the emission's own:
  its `directivity()` (written for a scene file by
  `acoustic_spaces.directivity.write_directivity_csv`) and its
  `sound_power_db(density, sound_speed)`, which the flyover demo does
  ([a helicopter flying over](demos.md#a-helicopter-flying-over)). A tone sent louder ahead (1 + 0.8 cos θ in amplitude) and
  turned in phase with the height follows the exact field within 0.11 dB in level at
  30 m/s and 0.42 dB for the train at 4 kHz, as one signal does there, with a cardioid
  directivity on the source or without (see
  [Validation](validation.md#moving-sources-against-the-exact-field)). An emission
  measured or computed in a moving source's frame may carry its convective
  amplification.
- The walker must go slower than sound, and the wave solver's bands do not follow a
  walk with Doppler: with `simulation.wave.enabled` it stops with an error.

The **pass-by levels** come with it: the walkthrough of a walk with Doppler prints the
listener's L_Amax and when it is reached, the sound exposure level (SEL) and L_Aeq over
the pass-by, and `--levels CSV` (for any walk) saves the level over time, A-weighted
and in each octave band. Each is the level of the source's sound power at a place, as
`simulate` gives it, at the time its sound arrives, so its maximum and its exposure are
those of noise mapping's quasi-static source. In a band where a place's G is undefined,
as where the few rays that reach a far place carry no energy in it, the place's level
is that of the energy its response holds, as outdoors (-inf where none arrives).
Through the walk's sound over grass, each band's energy came within 0.25 dB of them,
0.8 dB in the band of the ground's dip, where the band levels are averages over the
band and the sound is read through band filters, whose skirts fill a dip (see
[Validation](validation.md#moving-sources-against-the-exact-field)).

```python
from acoustic_spaces.walkthrough import pass_by

walk = simulate_walk(room)  # the car, with doppler: true
sound = render_walk(walk, {"car": (engine, 48_000)})
passing = pass_by(walk)  # passing.max_a, .exposure_a, .equivalent_a, .levels_a

# Or the sound it sends each way: signals that do not correlate (here a tone's sine
# and cosine), each with an amplitude per direction on a grid of elevations from -90
# to 90 degrees and azimuths from 0 (ahead) up to 360, turning towards the left.
from acoustic_spaces.emission import Mixture

sent = Mixture(signals, weights, 48_000)  # (K, samples), (elevations, azimuths, K)
sound = render_walk(walk, {"car": sent})
```

### Head tracking

With headphones and a head tracker, a binaural rendering stays put in the room while the
head turns. `play --head-tracking` takes an Ambisonics response (`--output
ambisonics:3` writes `.ambix3.wav`), runs the dry signal through it in real time, turns
the sound field against the head once per block and decodes it to the ears:

```sh
acoustic-spaces simulate examples/church/scene.yaml --out church --output ambisonics:3
acoustic-spaces play church/organ__nave-middle.ambix3.wav organ.wav --head-tracking
```

The head's orientation comes from OSC messages on UDP port 9000 (`--osc-port`) of this
computer (`--osc-host 0.0.0.0` takes them from the network too, as from a phone): `/ypr`
with yaw, pitch and roll in degrees, `/quaternion` (or `/quaternions`, `/quat`) with w,
x, y and z, or `/yaw`, `/pitch` and `/roll` on their own, under any prefix
(`/SceneRotator/ypr` too). Yaw turns left, pitch looks up and roll tilts the head to the
right; `--invert yaw,pitch` flips the axes a tracker counts the other way round. Each
turn crossfades over one block, so the head moves without clicks, and the latency is
still the block.

The decoder fits the HRTF (a SOFA file with `--hrtf`, or the spherical head model) with
the spherical harmonics up to the response's order. Above 600 Hz per order it fits the
magnitudes only (MagLS, after Schörkhuber, Zaunschirm and Höldrich, 2018): with the
spherical head at third order, every octave from 250 Hz to 16 kHz keeps its level within
0.1 dB, and the level difference between the ears is within 0.5 dB rms over all
directions, where a plain least-squares fit (`--decoder ls`) is 16 dB short at 16 kHz.

In the editor, tick **Head tracking** in the Results panel: the yaw and pitch sliders
turn the head by hand, a tracker sending to the **OSC port** turns it as you move
(**Centre** takes your current orientation as straight ahead), and **Listen** plays
through the head-tracked rendering.

From Python:

```python
from acoustic_spaces.headtracking import HeadTrackedBinaural, OscReceiver
from acoustic_spaces.render import play
from acoustic_spaces.spatial import Ambisonics

ambix = response.spatial_impulse_response(Ambisonics(order=3), 48_000)
renderer = HeadTrackedBinaural(ambix, 48_000, block=256)
renderer.orientation.look(yaw=30)  # by hand, or let a tracker turn it:
with OscReceiver(renderer.orientation, port=9000):
    play(renderer, 48_000, speech_48k)
```
