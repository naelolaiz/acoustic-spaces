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
  speakers either side play, so arrivals there pan as in 5.0. Named layouts carry their WAV channel mask; the LFE channel of the
  `.1` layouts stays silent, leaving bass management to the playback system.
- **Binaural** gives every reflection and each late direction an HRTF interpolated
  between the measured directions: the measured directions are triangulated as for
  VBAP, and the responses at the corners of the triangle a direction falls in are
  aligned at their onsets, mixed with its barycentric weights and delayed by the same
  mix of their onsets, so the interaural time difference follows the direction smoothly
  even with a coarse set. Without a SOFA file it uses Brown and Duda's spherical head
  model: interaural time and level differences, but no pinna cues. `read_sofa` reads
  SimpleFreeFieldHRIR files, which most HRTF databases (for example those listed by
  [sofaconventions.org](https://www.sofaconventions.org)) provide; it needs h5py.

Levels match the mono response: an arrival from straight ahead has the same height in
W, in the speaker it is panned to, and, below the head's influence, at both ears.


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
levels the simulation gives each source. Dry signals are mixed down to mono, as
sources are points, and resampled to the response's rate. The results are 32-bit float
WAV files; `--normalize` scales them to a -1 dBFS peak.

`play` convolves in real time on the sound card (it needs the `measure` extra). The
compiled convolver adds no latency beyond one `--block` of frames (256 by default, 5.3
ms at 48 kHz), and nothing allocates memory while it runs. It cuts the response into
partitions that grow along it, from the block size up to 8192 frames, and spreads the
work of the long ones evenly over the blocks in between (non-uniform partitioning), so
long and multichannel responses stay cheap: a 3 s, 16-channel third-order Ambisonics
response in blocks of 128 frames takes about 0.3 ms of the 2.7 ms each block lasts, on
one current x86-64 core, where uniform partitions took 3.6 ms. By default the response
is scaled to unit energy so the result plays about as loud as the dry signal;
`--absolute` keeps the simulated level.

From Python:

```python
from acoustic_spaces.render import Convolver, auralize, convolve, play, resample
from acoustic_spaces.spatial import Binaural

mix = auralize(result, {"talker": (speech, 44_100)}, "seat-a", Binaural())  # (samples, 2)
wet = convolve(speech_48k, binaural)  # offline, float64
convolver = Convolver(binaural, block=256)  # block by block, float32
out = convolver.process(next_block)  # (256, 2)
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
  [`orchestra` example](#an-orchestra-in-the-concert-hall).
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

### An orchestra in the concert hall

[![Beethoven 8 in a virtual concert hall: the orchestra on the stage of the concert hall and a seat in the middle of the stalls, with the first reflections from a violin to it](images/orchestra_video.png)](https://www.youtube.com/watch?v=c0-5F-iNhIE)

[The video](https://www.youtube.com/watch?v=c0-5F-iNhIE) plays the first movement of the
Beethoven recording above as heard from the middle of the stalls of the concert hall
example, binaurally (use headphones). Each of the 61 tracks is one source on the stage,
in German seating: violins on both sides of the conductor, the winds and brass on
risers, every player facing the conductor with a directivity that narrows above 1 kHz,
horn bells pointing back. The `orchestra` example is that scene, with four seats from
the conductor's podium to the back of the hall, and the scripts that made the video:

```sh
acoustic-spaces examples --copy . orchestra && cd orchestra
unzip movement1.zip -d movement1  # from https://doi.org/10.14279/depositonce-6729.2
acoustic-spaces simulate scene.yaml --out results
bash auralize.sh results movement1/anechoic stalls-middle binaural
python make_images.py --seat stalls-middle --movement 1  # cover, seating, thumbnail
bash make_video.sh beethoven8_movement1_stalls-middle.binaural.wav  # needs ffmpeg
```

`auralize.sh` maps each source to its track and calls `auralize`; the mix is
normalized to -1 dBFS. All sources have the same power, so the balance between the
instruments is the recording's own. `make_images.py` draws the pictures with Pillow
(`viz` extra) and `make_video.sh` puts a scrolling spectrogram of the mix under the
cover. Movements 2 and 4 work the same way.

### Walkthroughs

A listener or a source with a [walk](scenes.md#walks) can be heard as it goes. The
`gallery_walk` example has a visitor walk away from a guide, down the long arm of the
L-shaped gallery and round the corner, while the guide talks (any dry speech recording
does as `speech.wav`; see [dry recordings to try](#dry-recordings-to-try)):

```sh
acoustic-spaces examples --copy . gallery_walk && cd gallery_walk
acoustic-spaces walkthrough scene.yaml guide=speech.wav --out walk.wav --video walk.mp4
```

- Responses are simulated every `--step` along the path (25 cm by default, 101 places
  for the gallery's 25 m). The early sound, which changes quickly from place to place,
  is simulated at every one of them; the late field changes slowly, so the late part,
  by far the costliest to trace, is traced only every `--late-step` (2 m by default,
  14 places), and the places in between mix the late parts of the two on either side
  in proportion to how near they are. Along the gallery, late parts mixed from places
  4 m apart, twice the default, came within 0.3 dB on average of those traced at the
  places themselves, band by band, and within 1.3 dB next to the corner, where the
  field changes fastest (with 10,000 rays per band, whose own noise is part of that).
  The gallery's walk takes about five minutes on 4 cores, six times as long as one
  response.
- The dry signals play through the response of the place the walker is nearest to,
  with a `--crossfade` (a raised cosine, 50 ms by default) half-way between two places,
  so the sound changes smoothly as it walks: the crossfades' weights always add up to
  one, and one response at every place gives exactly the convolution with it. Through
  the gallery's walk, a 440 Hz tone has at least 75 dB less energy above 2 kHz than in
  the tone in every 10 ms, much as through a single response; linear crossfades left
  only 60 dB at their corners. Every place's late part is drawn from the same random
  sequence, so the tails that cross fade differ only in their level and decay, not in
  their noise. Once the walk is over, the walker stays where it ends for as long as the
  signals last.
- A walking listener hears every source; a walking source is heard at `--listener`,
  with the other sources where they stand. `--format` takes the formats of `simulate
  --output` (by default the scene's first output) and `--video` saves the room, the
  path and the walker going along it, with an arrow and its yaw for the way it faces,
  as a GIF or an MP4 (both with the viz extra) at `--fps`. An MP4 carries the sound
  when the format has one or two channels (mono, stereo, binaural) and then lasts as
  long as the sound; with more channels, play the WAV alongside it.
- A walker faces the way it goes (unless its walk says `face_forward: false`) and
  turns through each corner at 90 degrees per second. The gallery's visitor has the
  guide behind it all down the long arm, and at the corner turns over a second towards
  the short arm, which brings the guide round to its right. The default binaural head
  is a sphere, which tells left from right but hardly front from behind (it has no
  outer ears), so the guide behind sounds much as one ahead would; a measured HRTF
  (`--hrtf`) tells them apart.
- There is no Doppler shift: at walking speed it is under half a percent.

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
spherical head at third order, every octave from 125 Hz to 16 kHz keeps its level within
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
