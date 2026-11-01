# Demos

Five demos run recordings through example scenes, mix them at the scenes' listeners and
make a video of the result, a sixth drives a car, a truck, a bus, a motorcycle, a
police car with its siren on or a fire engine with its siren and its air horn past the
listeners of two outdoor examples, a tram down the street of one of them and a train
past those of a third, a seventh flies a helicopter over those of a fourth, an
eighth flies it over the stadium's concert, a ninth plays road works in the street and
a tenth flies a jet taking off from an aerodrome and landing on it.
Each shows one space and one kind of sound: a symphony orchestra
on the stage of a shoebox concert hall, a choir and an organ in a reverberant stone
church, a band with its sound system and stage monitors in a stadium, the same band's
song carried across a lake on a summer evening, a guide talking to a visitor who walks
away from them through a gallery and then turns their head, a vehicle passing on a road
across a meadow and along a street and a train on a railway across a meadow, its pitch
falling as it goes by, a helicopter
flying over a meadow and low over the stadium during the song, a breaker, a
generator and a dumper reversing with its alarm on at road works in the street, and a
twin jet climbing away from a runway and coming down to it. None of them
stands for what the simulator does elsewhere; the [examples](examples.md) cover far
more, and any scene makes a demo of its own with the recordings its sources are given
([a demo of any scene](#a-demo-of-any-scene)).

The scripts come with the examples: `acoustic-spaces examples --copy . NAME` copies them
with the scene. The recordings do not: they are other people's work under their own
licences, named below with each demo, and the scripts download them from where their
authors publish them. The videos need [ffmpeg](https://ffmpeg.org)
on the PATH and the pictures Pillow (the `viz` extra brings it). The mixes are binaural:
use headphones. They go through the spherical head model unless every step and the
`simulate` command get `--hrtf me.sofa`, a SOFA file of your own or measured HRTFs (the
`simulate` command a script prints carries it; the `sofa` extra reads the file). The
videos are in one [playlist](https://www.youtube.com/playlist?list=PLWz5JDREfod8).

## Running a demo

Every demo is a script of steps: it writes its scene, simulates it, mixes what each
listener hears and puts the video together. A step is run by name, and the steps of a
demo are listed by `python NAME_demo.py --help`:

```sh
python church_demo.py music    # the recordings it plays, downloaded
python church_demo.py scene    # the scene the demo simulates
python church_demo.py simulate # acoustic-spaces simulate on that scene
python church_demo.py all      # every step in order, leaving out what is done already
```

`all` is the short way through: it skips the steps whose result is in the work folder
already (the scene file, the simulation's results, each stop's mix), so a run that
stopped halfway carries on where it left off, and a change to one step is seen in the
video without simulating again. `--again` mixes the stops afresh even where their files
exist, `--work FOLDER` puts everything somewhere else, and `--no-mp4` writes the audio
and the stills without calling ffmpeg. Where a step needs an earlier one, it says which
command to run.

`--listeners a,b` has a demo play those listeners alone: it mixes them and its video
tours them, leaving the others out. Every demo with several listeners takes it, with the
names the scene gives them (a name the demo cannot play is refused with the list of
those it can). The simulation still takes every listener in use, so another choice later
needs no new simulation, except in the pass-by demos, which simulate the vehicle along
its walk at each listener they play (a pass made already is kept). The demo of any scene
and the pass-by demos visit them in the order given; the others keep their own tour's
order: the church's tour keeps its times, a place left out giving its time to the one
before; the stadium's and the lake's stops there keep their phrases, the last running to
the end; the orchestra plays each movement at the seats of its tour among them, or at
all of them when its tour has none. The gallery walk hears its visitor alone and takes
no `--listeners`.

`--input SOURCE=WAV`, once for each source, has a source play a recording of yours
instead of what the demo gives it: `python lakeside_demo.py render --input
subs=bass.wav`, or a singer of the choir (`--input singer_f1=take.wav`), a player of the
orchestra (`--input fl1=flute.wav`, in every movement), any source of the stadium. The
file is mixed down to mono, brought to the gated RMS of the feed it replaces, so the
source plays as loud as the demo has it, and cut or padded with silence to that feed's
length, so the tour's times hold; a name the demo has no source for is refused with
the list of them. The mixes are made again when the inputs change. The credit line on
the stills names the files that replaced a recording (and only them when every source
has one); the captions go on describing the demo's own music. The walk through the
gallery takes `--input guide=talk.wav`, played as it is: the video's one gain sets how
loud it is heard, and the captions' level and STI stay the scene's voice; the pass-by
demos take `--input car=engine.wav`, a dry recording played as the car's sound.

In the editor, **Simulation > Run demo…** does the same for the open scene: *This
scene* (below), and for an example's scene, or a copy of one, the example's own demos
and its scripts that are no demos. The window shows the options that change what the
demo plays, at the command line's defaults, the file each source plays, the listeners
to play and the walkers that move; **Advanced** holds the step, the paths and the
command line, to repeat the run in a terminal. A line above **Run** says what the run
downloads and how much. A scene with changes is saved first, as the demo reads its
file.

Every still says how its sound was made, at the bottom: the version, the early
reflections (traced or image sources, their order, diffraction) and the late sound with
its rays, as the simulation ran them, the wave solver's bands when it ran, outdoors in
moving air the bent paths and the parabolic equation's bands, the HRTFs the mixes went
through (or the spherical head model) and the sample rate; and under that the command
that makes the video again, with the options it was given and paths by their names (the
song, the HRTFs).

## A demo of any scene

Any scene makes a demo: each source plays a recording, every listener is heard in
turn, and a video tours them. A source names what it plays in the scene file, or with
**Plays** on its page in the editor (**Choose WAV…**; **Silent** takes it away):

```yaml
sources:
  - name: talker
    position: [2.0, 3.0, 1.5]
    level_db: 78
    spectrum: speech_male
    signal: sounds/talk.wav   # a dry recording, relative to the scene's folder
```

A sound system plays one recording through all its loudspeakers instead: the scene
names it once, as its `recording` (**Recording** on the scene's page in the editor),
and each loudspeaker that plays it gets a `channel` (**Channel** on its page): `left`
or `right`, or `mono`, both mixed down, as subwoofers and loudspeakers in the middle
play. The recording is a file of yours, such as a band's stereo mix (WAV, or anything
ffmpeg decodes), or the name of a free one, downloaded the first time a demo plays it:

```yaml
recording: spicy_funk_cake    # or a file of yours: music/our_band.flac
sources:
  - name: house_left          # on the audience's left
    position: [-40.0, -19.0, 14.0]
    channel: left
  - name: subs
    position: [-41.0, 0.0, 0.6]
    channel: mono
```

| Free recording | What it is | Licence | Download |
| --- | --- | --- | --- |
| `spicy_funk_cake` | "Spicy Funk Cake" by the Rhythmussportgruppe, a nine-piece funk band, its stereo mix ([Zenodo](https://doi.org/10.5281/zenodo.3601032)) | CC BY 4.0 | 61 MB |
| `great_british_summer` | "Great British Summer" by Tom McKenzie, its stereo master ([Zenodo](https://doi.org/10.5281/zenodo.19683001)) | CC BY 4.0 | 50 MB |
| `potassium` | "Potassium" by Software-Entwicklungskit, its stereo master ([album](https://github.com/ichbinsoftware/everythingisfree)) | CC0 1.0 | 6 MB |
| `speech_male` | a man talking, anechoic: talker M4 of the ACE challenge corpus ([Zenodo](https://doi.org/10.5281/zenodo.6257551)) | CC BY 4.0 | 16 MB |
| `speech_female` | a woman talking, anechoic: talker F2 of the same corpus | CC BY 4.0 | 49 MB |

They are the songs the lakeside demo plays and the talks of the gallery's. A free
recording is kept as one WAV file in the user's data folder (`acoustic-spaces/recordings`
under `~/.local/share` on Linux, `~/Library/Application Support` on macOS,
`%LOCALAPPDATA%` on Windows; `ACOUSTIC_SPACES_RECORDINGS` names another), so every
scene that names it plays the same file and none downloads it again. The stills credit
it as its licence asks unless `--credit` says otherwise. A source's own `signal` wins
over the recording, and its `channel` then picks a channel of that file; a source with
neither stays silent. The examples with a stage, a sound system or a talker name a free
recording and give their loudspeakers or talker a channel, so their demo plays out of
the box: concert_stage, stadium_concert, lakeside_concert (its stage; ticked, its PA
group as lakeside_demo.py feeds it), open_air_stage and studio a song; town_hall_pa,
lecture_hall, leaky_classroom, shoebox, bim_office, material_libraries and gallery the
male talk. A stereo mix has none of the parts that the monitors or a band on the stage
play, so they stay silent; the stadium demo plays its song's stems through them.

```sh
acoustic-spaces demo scene.yaml                 # every step, into demo/ beside the scene
acoustic-spaces demo scene.yaml recording       # download the scene's free recording
acoustic-spaces demo scene.yaml feeds           # one step: feeds, simulate, render, video
acoustic-spaces demo scene.yaml --seconds 15 --listeners front,back --hrtf me.sofa
acoustic-spaces demo scene.yaml --moving none   # every walker standing where it starts
```

- **recording**: downloads the free recording the scene names, unless it is there or
  no source plays it; the editor's Run demo window says so above Run, with its size.
- **feeds**: each source's file (its `signal`, or the scene's `recording`), at the
  scene's sample rate, its `channel` or mixed down to mono. Each file is brought to one
  gated RMS (unweighted, its channels mixed down) and every channel taken out of it
  keeps that gain, so a recording's stereo balance stays and how loud each source plays
  against the others is its sound power in the scene (`level_db`, with its feed's gain
  and equalizer), as the simulation says, and not how loud the recording happens to
  be. Sources with neither stay silent and are listed as such. `--input
  SOURCE=WAV` plays another file instead of a source's signal or the recording, without
  changing the scene (the source's `channel` is taken out of it in the same way), and
  gives a silent source one.
- **simulate**: `acoustic-spaces simulate` on the scene, its binaural output through the
  HRTFs the scene's binaural output names, `--hrtf` instead, or the spherical head
  model when neither names any. A scene changed since its results is simulated again.
- **render**: every listener in use (or those `--listeners` names, in its order;
  `--tour`, its earlier name, still works) hearing every source that plays, with one
  gain for all, so that the listeners' differences in level are heard as they are.
- **video**: `tour.wav` hears each listener for `--seconds` (20 by default) while the
  recordings run on, crossfading between them; a recording too short for every listener
  shares its length out. Each stop is turned up to the loudness (K-weighted) of the
  loudest, by as much as the still says, the only step that is not the simulation's. The
  stills show the scene from above, drawn from its own model, with its sources (those
  that play in colour, with a ray to the listener), the listeners and the one of the
  moment as a head, and beside it the A-weighted level there of the sources that play,
  as simulated, when their sound arrives, the reverberation time T30 at mid frequencies
  (the mean of the 500 Hz and 1 kHz bands) and the nearest source that plays. `--credit`
  puts a line about the recordings on the stills (who made them, their licence); without
  it the line gives the free recording's credit, when one plays, and says which source
  plays which other file.

**Movement.** Sources and listeners whose [walk](scenes.md#walks) is on move along it
while the recordings play, all on one clock from their start, as `walkthrough --all`
plays them ([walkthroughs](spatial-audio.md#walkthroughs)). Then **render** simulates
the scene with every walker at each moment of its walk, in one simulation, and mixes
every listener hearing the walkers go by, or walking itself, with one gain for all; the
still simulation stays for what the stills say where nothing moves. The video is a
moving picture: the walks dotted, every walker where it is at that moment, and the
level of the sources that play as simulated at that moment; the arrival time is left
out where something moves, and T30 is the one where the walks start. A stop hears its
listener where the clock has it then, so `--listeners` puts a walker first to hear its
walk from the start. `--moving` picks the walks of a run without changing the scene:
`--moving visitor,guide` moves those two and leaves every other walker standing where it
starts, `--moving none` keeps them all still, so one scene serves both. A walk with
Doppler (a car, a train) moves alone: every listener hears each early path through its
own delay as it changes, so the pitch shifts as it passes, and the late sound
crossfaded from place to place
([moving sources and Doppler](spatial-audio.md#moving-sources-and-doppler)), all from
the one simulation of the moving run, as the editor's **Render movement…** plays it.
With other walks on as well, the demo says to leave them standing.

Everything goes to `demo/` beside the scene (`--work` elsewhere): the feeds, the
results, the mixes in `seats/` (in `moving/` when something moves), `tour.wav`, the
stills, `thumbnail.png` and `tour.mp4`.
In Python the demo is `acoustic_spaces.demos.scene_demo`
(`python -m acoustic_spaces.demos.scene_demo all --scene scene.yaml`).

## An orchestra in the concert hall

[![Beethoven 8 in a virtual concert hall: the orchestra on the stage of the concert hall and a seat in the middle of the stalls, with the first reflections from a violin to it](images/orchestra_video.png)](https://www.youtube.com/watch?v=c0-5F-iNhIE)

`orchestra_demo.py` plays the three movements the TU Berlin's anechoic recording of
Beethoven's 8th symphony has (one of the [dry recordings to
try](spatial-audio.md#dry-recordings-to-try); 70 to 85 s of each) in the concert hall
example, binaurally (use headphones), and makes a video of about four minutes that tours
its four seats through them: the first
movement from the middle of the stalls, the front rows and the conductor's podium, the
second from the back of the hall and the middle, the finale from the podium, the front,
the back and the middle again. Each of the 61 tracks is one source on the stage, in
German seating: violins on both sides of the conductor, the winds and brass on risers,
every player facing the conductor with a directivity that narrows above 1 kHz, horn
bells pointing back. The `orchestra` example is that scene, with the four seats from the
conductor's podium to the back of the hall.

```sh
acoustic-spaces examples --copy . orchestra && cd orchestra
python orchestra_demo.py tracks   # demo/tracks: the 61 tracks of each movement (2.0 GB)
acoustic-spaces simulate scene.yaml --out results
python orchestra_demo.py render   # demo/seats: each stop of the tour, binaural (~1 min each)
python orchestra_demo.py video    # demo/tour.wav, demo/stills, demo/tour.mp4, demo/thumbnail.png
```

Options: `--tour 1:stalls-middle,1:conductor,2:stalls-rear` (the seats of a movement take
equal shares of it), `--listeners conductor,stalls-rear` (each movement at the seats of
its tour among them, or at all of them when it has none of them), `--zips FOLDER` for a folder with zips downloaded before,
`--movements 1,2` for `tracks`, `--format` for another output the simulation wrote,
`--no-mp4` for the audio and stills only. `demo/chapters.txt` lists the stops' times for
a video description. `all` runs tracks, the simulation, render and video, leaving out
what is there already.

- **The cuts.** Within a movement the seat changes at the quietest moment within 10 s of
  each equal share's end (the lowest half-second RMS of the first seat's mix), so the
  cuts fall between phrases, with a 1 s crossfade. Between movements the hall's
  reverberation dies away (the responses run 3.5 s) and there is a pause of 3 s. Every
  seat's mix is shifted earlier by the time the orchestra's sound takes to reach it (the
  peak of the energy of its responses: 10 ms on the podium, 85 ms at the back), so the
  music runs on across a cut.
- **One gain for all.** All stops keep one gain (the loudest peak at -1 dBFS): the hall's
  differences are heard as the simulation gives them, from the podium, 2 m from the
  first violins, to the back row, and each still says how far below the loudest stop it
  is. Every source has the same power, so the balance between the instruments is the
  recording's own.
- **The pictures.** `make_images.py` draws the cover (the hall from above with the
  orchestra, the seat and the first reflections from the leader to it, the sections and
  their numbers), the seating plan and the thumbnail with Pillow (`viz` extra); the
  demo adds the stop's facts and a bar across the top with the tour, the movements'
  numerals and the seats, the current stop bright.

One seat and one movement at a time still work as before: `auralize.sh` maps each source
to its track and calls `auralize` (the mix normalized to -1 dBFS), `make_images.py
--seat stalls-middle --movement 1` draws the pictures and `make_video.sh` puts a
scrolling spectrogram of the mix under the cover.

**Recording.** Böhm, Ackermann and Weinzierl, *Beethoven, Symphony No. 8*, TU Berlin,
<https://doi.org/10.14279/depositonce-6729.2>, CC BY 4.0: movements 1, 2 and 4, 61 mono
tracks each, in a zip with the movement's mix (590 to 740 MB). `tracks` reads the 61 WAV
files of each out of its zip on the TU Berlin's server by HTTP range requests, without
keeping the zip: 640, 600 and 740 MB, 2.0 GB for the three (nearly the whole zip, as the
blocks of 32 MiB it reads of the last tracks hold the mix), 2.6 GB on disk. It reads
them out of the zip, or the folder it was unpacked into (`movementN/anechoic`), in
`--zips` instead, the example's folder by default. Tracks there already are kept, so a
download cut short goes on where it stopped.

## A choir and an organ in the church

![A frame of the church video: the church from above and from the side, with the choir in front of the altar, the organ in the gallery and the conductor facing the choir, a line from each singer to the conductor, the choir close up, the tour and a spectrogram of the sound](images/church_demo.png)

`church_demo.py` puts a choir of 19 real singers in front of the altar of the `church`
example and the organ in the gallery over the entrance, and plays the round "Dona nobis
pacem" three times: the organ alone on full stops, the choir alone, then the choir with
the organ on quiet stops, slowing into a held last chord and the church's reverberation
(1:26 in all). It mixes the piece at six places and makes a video that tours them: under
the organ at the back, along the south wall, the front pews, the conductor, the back row
of the choir and the middle of the nave. The choir, the organ's pipes and those places
are the example's: its Choir, Organ pipes and Choir concert groups, left out of the
church's own simulations until ticked in the editor, and its listeners `nave-back` and
`nave-middle`.

```sh
acoustic-spaces examples --copy . church && cd church
python church_demo.py music     # 19 takes, their annotations, 27 organ samples (127 MB)
python church_demo.py scene     # scene_choir.yaml: the church with the choir and the organ
acoustic-spaces simulate scene_choir.yaml --out results_choir --output binaural   # 10-20 min on 4 cores
python church_demo.py feeds     # demo/feeds: one WAV per singer and per side of the organ
python church_demo.py render    # demo/seats: the whole piece at each listener
python church_demo.py video     # demo/tour.wav, demo/stills, demo/tour.mp4, demo/thumbnail.png
```

Options: `--tour nave-back:0,conductor:25,...` (listener:start in seconds),
`--listeners nave-back,conductor` (those places alone, in `render` and `video`), `--format`
for another output the simulation wrote (`ambix3`, `5.0`, ...; the video takes its first
two channels), `--no-mp4` for the audio
and stills only. The scene and its simulation go beside the script (`scene_choir.yaml`,
`results_choir`), the rest to `./demo` (`--work`). `all` runs every step in turn,
leaving out what is done already.

- **The choir.** Each singer sang the tune alone in a recording booth, at their own tempo
  and a little off pitch here and there. `feeds` moves each take towards the tune (each
  singer's own offset fully, 60 % of each note's error) and warps it in time so that every
  note starts on the beat, from the dataset's note annotations; after that the voices
  start within about 40 ms of each other and sit about 16 cents from the tune on average,
  as a real choir does. The women stand in the front row, the men on a step behind them,
  0.6 m apart, facing the nave (sources with a voice-like pattern: omni in the bass,
  cardioid from 2 kHz), 88 dB sound power each.
- **The organ.** A four-part harmonisation of the tune, written for the demo, played from
  a sampled church organ: an open 8' with the same rank an octave up as a 4' and the pedal
  with its 16' for verse 1, quiet stops 11 dB down for verse 3. As in a real organ, each
  division's pipes stand in two halves, C side and C# side, so the organ is four sources
  across the gallery, 9.5 m up for the manual and 8.6 m for the pedal towers, facing the
  nave (105 dB for each half of the manual, 102 dB for each pedal tower).
- **How loud.** The responses are per unit of sound power and `auralize` scales each
  source by its `level_db`; the feeds are brought to one gated RMS where they play
  loudest, so the balance between the singers and the organ is the scene's.
- **One gain for all listeners.** `render` scales the six mixes by one gain (the loudest
  peak at -1 dBFS), so the level differences between places are the simulation's.
- **The cuts** fall where the music breathes (the rest in the middle of the tune, the gaps
  between verses), with a 1 s crossfade. Time is the performers': a listener far from the
  choir hears it up to 80 ms later than one near it, as in the church.
- **The pictures** draw the church from above and from the side, each listener with an
  arrow the way they face and the current one with L and R at its ears, the sources as
  dots with see-through cones the way they face, and lines from each source playing to
  the listener, brighter where it is louder. The caption gives the clarity C80 there.

**Limits.** A "virtual choir": the takes were recorded separately, so the singers do not
hear each other or the room; their timing and tuning are evened out by the script, not by
listening. The organ samples hold a little of the room they were recorded in, and the
organ is four points, not a case full of pipes. The model is the example's: the floor
plan raised 18 m, with no gallery floor under the organ and no columns or furniture;
`ir_length_s` is raised to 9 s, as the bass rings for nearly ten. Binaural is through the
spherical head model unless `--hrtf` names a SOFA file.

**Recordings and licences.** `music` writes these credits to `demo/music/SOURCES.txt`.

- The singers: Wilkins, Seetharaman, Wahl and Pardo, **VocalSet: A Singing Voice
  Dataset** (ISMIR 2018), <https://doi.org/10.5281/zenodo.1442513>, **CC BY 4.0**. The
  straight-tone takes of "Dona nobis pacem" of 19 of its 20 professional singers, close to
  a cardioid microphone in a booth (mono, 44.1 kHz). The countertenor is left out: the
  annotation of that take belongs to a longer take than the one in the archive.
- Where the notes are: Faghih and Timoney, **Annotated-VocalSet** (Applied Sciences
  12(18), 2022), <https://doi.org/10.5281/zenodo.7061507>, **CC BY 4.0**: the start, end
  and mean F0 of each sung note. Its ground truth for the fourth note says F4; the score
  and the singers have G4, which the demo uses. Both archives are read by HTTP range
  requests, so only the parts holding the files needed (69 MB) are downloaded, not the
  2.5 GB.
- The organ: the Versilian Community Sample Library's pipe organ, sampled by Simon
  Dalzell of Ivy Audio, Versilian Studios, <https://github.com/sgossner/VCSL> (pinned
  commit), **CC0 1.0**, credited as its notes encourage. Sampled every minor third; each
  note comes from the nearest sample, retuned by up to two semitones. The samples hold
  about a second of the room they were recorded in (30 dB down after 0.7 to 0.9 s); the
  release fade removes most of it.
- The music: "Dona nobis pacem" is a traditional round (public domain). The organ's
  harmonisation (`ORGAN` in `church_demo.py`) was written for the demo; use it freely.
- Changes, as CC BY asks to say: each take is retuned towards the tune by resampling,
  time-warped so that its notes start on the beat (WSOLA, which keeps the pitch), cut
  into two verses, brought to one level and mixed through the simulated church. The organ
  samples are evened out in level along the keyboard and cut at each key's release with a
  90 ms fade.

Whoever shares a render (the video on a video site, say) has to credit VocalSet, link
the licence and say that the takes were changed, for example:

> Singers: VocalSet by J. Wilkins, P. Seetharaman, A. Wahl, B. Pardo (CC BY 4.0,
> doi:10.5281/zenodo.1442513), with note annotations from Annotated-VocalSet by B. Faghih
> and J. Timoney (CC BY 4.0, doi:10.5281/zenodo.7061507); retuned, time-aligned and placed
> in a simulated church. Organ: VCSL pipe organ, sampled by Simon Dalzell (Ivy Audio),
> Versilian Studios (CC0). License: https://creativecommons.org/licenses/by/4.0/

No multitrack recording of a real choir could be used: Dagstuhl ChoirSet, the Choral
Singing Dataset, the ESMUC Choir Dataset and Cantoría are all for non-commercial use only
(CC BY-NC 4.0).

## A concert in the stadium

![A frame of the stadium video: the stadium from above with the sound system, the monitors and the band, the listener at the mix position facing the stage with a line from each loudspeaker, the crowd standing on the pitch and seated in the stands, the stage close up, the tour and a spectrogram of the sound](images/stadium_demo.png)

`stadium_demo.py` plays a song by a band on the stage of the `stadium_concert` example,
through its sound system and stage monitors, and mixes it at each of its 12 listeners.
The video plays the song once while the listener moves, on phrase boundaries, from the
mix position to the front of the pitch, onto the stage (singer, drummer, guitarist),
round the stands to the far end and back. It plays one of three songs (`--song`): "Spicy
Funk Cake", funk by a nine-piece band (3:15, ten of the listeners, as the song has eleven
phrases), or "Francium" (4:54, all twelve) or "Rubidium" (3:52), electronic songs by
Software-Entwicklungskit released with their stems.

```sh
acoustic-spaces examples --copy . stadium_concert && cd stadium_concert
python stadium_demo.py music     # the reference mix and 7 stems (WAV, 313 MB to download)
python stadium_demo.py band      # scene_band.yaml: the scene with the band on the stage
acoustic-spaces simulate scene_band.yaml --out results_band --output binaural   # ~3 min on 4 cores
python stadium_demo.py feeds     # demo/feeds: one WAV per loudspeaker and player
python stadium_demo.py render    # demo/seats: one binaural WAV per listener (~1 min each)
python stadium_demo.py video     # demo/tour.wav, demo/stills, demo/tour.mp4, demo/thumbnail.png
```

Options: `--song francium` or `rubidium` for `music` and `feeds` (with `--work
demo_francium` for a folder of its own), `--tour mix_position:1,singer:0.5,east_stand:1`
(listener:phrases of 8 bars, 0.5 for 4; the last one runs to the end), `--listeners
mix_position,east_stand` (those listeners alone, in `render` and `video`), `--format` for another output the simulation wrote, `--no-mp4` for the
audio and stills only. `all` runs music, feeds, render and video; `band` and the
simulation come before it.

A song of your own plays in place of the presets: `--music song.wav --bpm 120` for
`music` and `feeds` (and `all`), the master as a stereo file (WAV, or anything ffmpeg
decodes) and its tempo, which the tour's phrases of 8 bars go by. `--stem vocal=vox.wav
--stem drums=drums.wav`, once per stem (drums, bass, guitar, keys, brass, vocal,
backing), gives the stems the monitors' mixes and the band's players draw on; a mix
none of whose stems is given plays nothing, so without stems the monitors and the band
are silent, which `feeds` and the Run demo window's table say. `--downbeat 0.5` is the
time of the first downbeat, so that the cuts stay on the beat; `--credit "Music: …"`
the line the video shows, the file's name unless given. Nothing is downloaded; the
music goes to `demo/music/<file name>`, a folder of its own beside the presets', and
`--input` goes on working on top of it.

- **The band.** The players are the example's sources on the stage, in its Band group,
  which the concert's own simulations leave out until it is ticked; `band` takes them
  in: a drum kit beside the drummer (116 dB sound power, omni) playing the drums, the
  singer's voice (98 dB, subcardioid, facing the audience) playing the
  vocal, and an amplifier behind the guitarist (112 dB, cardioid, tilted up) playing the
  guitar, each from its own stem. On stage and near it the players
  are heard from where they stand; further out the sound system covers them.
- **Who plays what.** The sound system plays the song's stereo master: the arrays, fills,
  towers and delays on the audience's left (-y) take the left channel, those on its right
  (+y; the scene's `*_left`, named from the stage) the right one, the subwoofers both.
  The monitors play mixes of the stems: the singer's wedges the vocal over the backing
  vocals, the keys and a little of the guitar, bass and drums; the drum fill the drums
  and bass with the vocal, guitar and keys under them; the side fills a stage mix with
  the vocal 3 dB up and the drums 20 dB down (the kit is loud on stage by itself). A stem
  a song does not have (brass, hand percussion) is left out of the mixes that name it.
- **How loud.** `feeds` brings every signal to the same gated RMS and then scales it by
  its `level_db` against the loudest's: the main arrays 0 dB, subwoofers and towers -2,
  delays -4 and -6, out-fills -4, in-fills -10, side fills -12, front fills -18, wedges
  and drum fill -19. So the feed files are what each loudspeaker plays, and `render` runs `auralize`
  with `--equal-power`, so that the levels are applied once.
- **One gain for all listeners, then each stop turned up.** `render` scales the 12 mixes
  by one gain (the loudest peak at -1 dBFS), so the level differences between places in
  `demo/seats` are the simulation's: on stage, between the monitors and beside the drum
  kit, it is 7 to 20 dB(A) louder than in the audience, and the mix position is
  18 dB(A) below the singer's place. That is too much to hear both in one video, so the
  tour turns each stop up to the loudness of the loudest one and its caption gives the
  real level and how much it was turned up. Loudness is weighted as loudness meters
  weigh it (K, ITU-R BS.1770), which counts the subwoofers' bass that the A weighting of
  the levels all but leaves out: on the pitch the stops are turned up less than their
  dB(A) alone would say.
- **The cuts.** Every listener's mix is shifted earlier by the time the loudest sound
  takes to reach them (the peak of the energy of their responses, each weighted by how
  loud its source plays: 0.12 s at the mix position, 0.38 s at the far end), so the beat
  runs on across each 1 s crossfade; the caption gives that time. The stops last 8 bars
  from the song's first downbeat, each crossfade ending on a downbeat. "Francium" starts
  a beat into a bar and its intro is a bar short, so its phrases run from 14.8 s and the
  first stop takes the intro as well; it has nineteen phrases and a half, so five stops,
  each at a section's start, last two.
- **The pictures.** The stadium from above, with the loudspeakers as dots and cones the
  way they point, lines from each to the listener of the moment (brighter where louder),
  and the crowd where the scene has its audience: a dot for every ten or so people
  standing on the pitch (two on a square metre) and sitting in the stands (one and a
  half), and the empty stand behind the stage as rows of seats. A line under the plan
  gives how many stand and sit and how much of the sound at 1 kHz the crowd, the empty
  seats and the concrete take up, as the scene's materials have it.

**Limits.** The band is three point sources at typical sound powers, not measured
instruments; the bass, keys and brass on stage come only through the monitors and the
sound system, and there is no crowd noise. The reference mix and the stems are a
finished studio production, reverberation and all, so a little of the studio plays in
the stadium too. Binaural is through the spherical head model unless `--hrtf` names a
SOFA file. Some listeners stand nearer the arrays than their far-field patterns hold,
which the scene's arrays handle by adding up their boxes one by one (`near_field.py` in
the example puts numbers on it).

**Recordings and licences.** "Spicy Funk Cake" by the Rhythmussportgruppe, a nine-piece
funk band (guitar, drums, bass, keys, saxophone, two trumpets, trombone and voice;
composed by Niklas Dahlheimer, sung by Juliane Blum; 110 BPM, 3:15), recorded in 2017
for the sound database of the DEGA (German Acoustical Society) at Hochschule
Düsseldorf's studio, <https://doi.org/10.5281/zenodo.3601032>, **CC BY 4.0**. `music`
reads the reference mix and the seven processed stems (drums, bass, guitar, keys, brass,
vocal, shouts; stereo WAV, 24-bit, 48 kHz) out of the release's 3.5 GB archive by HTTP
range requests, so only those are downloaded: 313 MB (450 MB unpacked). The stems
start 58 ms before the reference mix; `feeds` lines them up with it by
cross-correlation.

"Francium" (127.75 BPM, 4:54) and "Rubidium" (132 BPM, 3:52) by Software-Entwicklungskit,
from the album *Everything is Free*, <https://github.com/ichbinsoftware/everythingisfree>,
**CC0 1.0**: electronic songs released with their stems. `music --song francium` downloads
the master and the 26 stems (M4A, 42 MB in all; "Rubidium" has 9, 43 MB) from the
album's repository at a fixed commit, decodes them with ffmpeg and sums the stems into
the groups the mixes use: drums, bass, guitar (the synth lead, which the guitarist's
amplifier plays), keys, vocal and backing vocals.

Either way the master plays through the sound system, the stems through the monitors
and the band, and the credit goes to `demo/music/<song>/SOURCES.txt`.

Whoever shares a render has to credit the musicians (and the DEGA), link the licence
and say what was changed, for example:

> Music: "Spicy Funk Cake" by Rhythmussportgruppe (composed by Niklas Dahlheimer),
> recorded for the DEGA at Hochschule Düsseldorf, CC BY 4.0, doi:10.5281/zenodo.3601032;
> the mix and its stems played through a simulated stadium sound system and stage and
> rendered binaurally. License: https://creativecommons.org/licenses/by/4.0/

Other band multitracks were left out: Cambridge-MT's are for education only, MedleyDB's
and MUSDB18's non-commercial or restricted, and the OnAir Music dataset's (CC BY-SA 4.0)
share-alike clause would bind the renders too.

## A concert across a lake

![A frame of the lake video: the lake from above with the wind's arrows and the weather zones, the stage on the lawn, the boat and the village across the water, the listener at a bedroom window there; under it a slice from the stage across the lake with the sound's rays bending down to the water, the weather's caption and a spectrogram of the sound](images/lake_demo.png)

`lakeside_demo.py` plays a song from the stage of the `lakeside_concert` example on a
summer evening and mixes it in the audience on the lawn, 140 m behind the stage, on a
boat in the middle of the lake and in the village across it, 550 m away, in a garden and
at a bedroom window. Every place is mixed twice: in the evening's weather and in still
air. The video tours them, and at the bedroom window plays the song in the evening's
weather, then in still air, then in the evening's again, so the difference the weather
makes is heard.

```sh
acoustic-spaces examples --copy . lakeside_concert && cd lakeside_concert
python lakeside_demo.py music    # the song (WAV, 56 MB), a signal for each loudspeaker
python lakeside_demo.py scenes   # scene_demo.yaml, and scene_still.yaml in still air
acoustic-spaces simulate scene_demo.yaml --out results_demo --output binaural    # ~30 s
acoustic-spaces simulate scene_still.yaml --out results_still --output binaural  # ~15 s
python lakeside_demo.py render   # demo/seats: every listener in both weathers
python lakeside_demo.py video    # demo/tour.wav, demo/stills, demo/tour.mp4, demo/thumbnail.png
```

Options: `--song great_british_summer` or `potassium` for `music` (below),
`--music song.wav --bpm 120` for a song of your own in their place (a stereo file, WAV
or anything ffmpeg decodes, and its tempo, which the tour's phrases of 8 bars go by;
nothing is downloaded, and the video credits the file by its name unless `--credit`
says what to show), `--tour bedroom_window:still:1,audience::2`
(listener:weather:phrases of 8 bars), `--listeners audience,bedroom_window` (those places
alone, in both weathers), `--last-phrases` (how long the last stop lasts, 0
to the end of the song; 2 by default), `--format` for another output the simulation
wrote, `--no-mp4` for the audio and stills only. `all` runs music, render and video;
`scenes` and the two simulations come before it.

- **The sound system.** `scenes` plays, in place of the example's single source (120 dB
  sound power), the example's PA group, left out of its own simulations until ticked: a
  system as a lawn concert hangs it: two line arrays of eight boxes
  (`lake_array.yaml`, a generic box's pattern in `box.csv`, both made with the stadium
  example's `make_model.py`), 6.5 m up either side of the stage, 124 dB each, aimed down
  the lawn and, box by box, from 5° down at the last rows, where the lawn meets the water,
  to 36° at the front rows, each added up box by box (`near_field`), playing the song's
  left and right channels; and a block of four cardioid subwoofers (`sub_array.yaml`,
  130 dB) on the ground at the stage front, 2 m nearer the audience and delayed 6 ms so
  that both arrive together there, playing both channels, rolled off above 125 Hz. The
  system is tuned at the audience as a system engineer tunes it (`eq_db`: the arrays' low
  mids up, their top octave up for the air): at 30 m every octave from 250 Hz to 4 kHz is
  within 1 dB of 1 kHz, 8 kHz 1.5 dB down and 63 and 125 Hz 2 dB up. `scenes` also takes
  in the example's listener 140 m behind the stage, upwind (`behind_stage`, left out as
  the system is), and lets the responses run 3.5 s, as the far shore hears the stage
  1.6 s late.
- **The weather** is the example's: a breeze from the park out over the lake, 2.5 m/s
  10 m up, and two weather zones, the lake's (air 17 °C at the water, a degree warmer
  every 10 m up, and a 3.5 m/s breeze) and the village's (a 1 m/s breeze among the
  houses). The breeze and the cool air bend the sound down to the water, which gives it
  back almost whole. `scene_still.yaml` is the same scene with no atmosphere.
- **What it gives.** At the bedroom window the evening's weather makes the song 7.1 dB(A)
  louder than still air would, in the garden 5.3 dB; behind the stage, upwind, it is
  0.5 dB quieter, and on the boat 1.0 dB quieter. The window gains most: what the
  weather bends down comes in off the water as well as straight, and the house's wall
  beside the window gives back each of those again.
- **How loud.** From the audience to the far shore the sound drops by 24 to 25 dB(A),
  too much to hear the far places in a video, so each stop is turned up to the
  audience's loudness and its caption says how much quieter the place really is, in
  dB(A). The loudness is K-weighted, as meters weigh it, not A-weighted: the far places
  and the back of the stage keep little but bass (the array boxes radiate every way below
  250 Hz and the water gives the low end back whole), which the A weighting all but
  leaves out, so matching dB(A) would put their bass far above the audience's. The window
  in still air keeps the gain of the window in the evening's weather, so the 7.1 dB is
  heard.
- **The cuts.** Each place's mix is shifted earlier by the time the loudest sound takes
  to reach it (the peak of the energy of its responses: 0.09 s in the audience, 1.66 s
  at the window), so that the beat runs on across each 1 s crossfade, every 8 bars.
- **The pictures** draw the lake from above with the wind's arrows, the weather zones and
  their weather, the arrays and the subwoofers with cones the way they aim, and the
  current listener with L and R at its ears; and, under it, a slice along the line from
  the stage across the lake, heights drawn five times larger, with the arrays hung from
  the roof and the wedge from their top box's aim to their bottom one's, the subwoofers
  on the ground, and rays from the arrays bent by the scene's own profiles of the speed of sound (traced
  by the script for the picture; the simulation traces its own) and masts with the wind's
  profile; in still air the rays go straight.

**Limits.** The arrays are a generic box's pattern, not a product's measured one; there
is no crowd, no wind noise and no other sound of the evening. The
weather holds still: the example has no turbulence, so the sound does not come and go
with the gusts as it does across a real lake. The song is a finished studio mix, played
as it is. Binaural is through the spherical head model unless `--hrtf` names a SOFA
file.

**Recordings and licences.** `music` writes the credit to `demo/music/song.json`, and
the video shows it.

- `spicy_funk_cake` (the default): "Spicy Funk Cake" by the Rhythmussportgruppe, the
  recording of the [stadium demo](#a-concert-in-the-stadium), **CC BY 4.0**; its reference
  mix (a 56 MB WAV file, 61 MB to download) is read out of the release's preview archive
  by HTTP range requests.
- `great_british_summer`: "Great British Summer" by Tom McKenzie, from the album
  *Extraordinary* in AlbumDB, <https://doi.org/10.5281/zenodo.19683001>, **CC BY 4.0**,
  read out of its archive of masters the same way.
- `potassium`: "Potassium" by Software-Entwicklungskit, from the album *Everything is
  Free*, <https://github.com/ichbinsoftware/everythingisfree>, **CC0 1.0**, decoded from
  M4A with ffmpeg.

A shared render of either CC BY song has to credit it, link the licence and say what was
changed, for example:

> Music: "Spicy Funk Cake" by Rhythmussportgruppe (composed by Niklas Dahlheimer),
> recorded for the DEGA at Hochschule Düsseldorf, CC BY 4.0, doi:10.5281/zenodo.3601032;
> played through a simulated outdoor sound system across a lake and rendered
> binaurally. License: https://creativecommons.org/licenses/by/4.0/

## A walk through the gallery

`walk_demo.py` is the smallest demo and the only one of speech: a guide stands in the
L-shaped gallery of the `gallery` example and talks while a visitor walks away
from them, down the long arm and round the corner, where the guide is out of sight and
only the sound bent round the corner arrives; then the visitor stands 4 m in front of
the guide and turns their head a quarter at a time, so the voice moves to the right
ear, behind, to the left ear and back in front. The video follows the visitor from
above, with the head and its ears, the guide, and a bar of the levels.

```sh
acoustic-spaces examples --copy . gallery && cd gallery
python walk_demo.py all   # or speech, scene, walk, turn, video one at a time
```

Options: `--talker M4` picks one of the corpus's 14 talkers, `--voice` how loud the
guide speaks (relaxed, normal, raised, loud), `--stand 16,4` where the visitor stands
for the turn (by default where the scene's `standing` listener does, left out of the
gallery's own simulations), `--turn` the angles it holds, `--hrtf me.sofa` a measured
head.

- **The walk** is the example's visitor's: responses every 25 cm along it
  (`acoustic-spaces walkthrough`), the talk played along them so that the voice changes
  as the visitor moves. The speech level at the ears falls from 56.0 dB(A) three metres
  in front of the guide to 48.7 dB(A) round the corner, and the speech transmission
  index with it, from 0.73 to 0.54: the gallery is quiet and bare, so what is lost is
  the direct sound, not the clarity. The gallery's two other listeners stand where they
  are and are left out. Edit the visitor's walk (the [Movement panel](editor.md), or the
  scene file) and `all` makes the scene and the walk again; with the walk turned off
  (**Moves**) the demo says there is no walk to play.
- **The turn** goes through the engine's head tracker, scripted rather than played
  live: the place's response in Ambisonics of order 7, the sound field turned against
  the head every 256 samples and decoded to the ears
  ([head tracking](spatial-audio.md#head-tracking)). It comes within half a decibel, in
  each ear and octave, of rendering the ears again at every angle.
- **Front and back.** Through the spherical head model the guide behind sounds much as
  the guide in front: a sphere has no outer ears. This is the demo where a measured
  HRTF (`--hrtf`, or [a set to download](spatial-audio.md#hrtf-sets-to-download)) is
  heard most plainly.
- **The voice** is IEC 60268-16's: the long-term spectrum of male speech at 60 dB(A) 1 m
  in front for a normal voice, with a voice's directivity, all round in the bass and
  towards the front in the treble. One gain plays the whole video, so the voice is as
  much quieter round the corner as the simulation says.

**Recording.** The ACE challenge corpus (Eaton, Gaubitch, Moore and Naylor, Imperial
College London), **CC BY 4.0**: an anechoic talk, a man describing his way to work
through Delft; `speech` reads the corpus's 148 MB archive as it comes, up to the
talk: 16 MB for that talker, up to 145 MB for another (`--talker`).

## Vehicles passing by

`passby_demo.py` comes with two outdoor examples, `roadside` and `street`: a vehicle
drives past and each listener hears one pass in turn, its pitch falling as it goes by.
The video follows the vehicle on the plan, with the level heard over the pass drawn
under it. `--vehicle` picks what goes past: in the roadside a passenger car (`car`) or
an articulated truck (`truck`), in the street the car (with `car_horn`, its horn, when
`--car` says), a city bus (`bus`), a motorcycle (`motorcycle`), a police car with its
siren on (`police_car`, the car itself, and `police_car_siren`), a fire engine with its
siren on and its air horn in blasts (`fire_engine`, the vehicle itself,
`fire_engine_siren` and `fire_engine_horn`) or a tram (`tram`, see [trains passing
by](#trains-passing-by)).

```sh
acoustic-spaces examples --copy . roadside && cd roadside   # or street
python passby_demo.py all --vehicle truck   # or sound, scene, walk, video one at a time
```

Options: `--vehicle car` what goes past (the car by default), `--speed 70` how fast it
drives (km/h; without it, the speed of its walk in the example's scene: 70 km/h for the
truck, 50 km/h for the others), `--rpm 2000` how fast its engine turns, for its made
sound (by default its own: the car's four cylinders at 2000 rpm, the truck's six at
1200, the bus's and the fire engine's six at 1500 and the motorcycle's two at 4000),
`--siren wail` the police car's siren (`wail`, `yelp` or `two_tone`), `--police-car
car_and_siren` what the police car plays (the car itself and its siren, or `siren`
alone), `--fire-engine truck_siren_and_horn` what the fire engine plays (all three,
`truck_and_siren`, `siren_and_horn` or `siren` alone), `--car car` what the street's car
plays (the car alone, `car_and_horn` with its horn, or `horn` alone), `--step 1` how far
apart the vehicle is simulated along its walk (m), `--hrtf me.sofa` a measured head,
`--input NAME=WAV` a dry recording of yours for the vehicle NAME (`police_car_siren=WAV`
for the police car's siren, `fire_engine_horn=WAV` for the fire engine's horn,
`car_horn=WAV` for the car's), `--listeners verge,yard_25m` the listeners it is
simulated and heard at (every one in use by default), in that order. The vehicle drives the length of its walk at that speed,
so the pass lasts the length divided by the speed: the 150 m of the roadside take 10.8 s
at 50 km/h, 7.7 s at 70 km/h. The **Run demo** window gives the three under its options
as they change, and ticks the vehicle (the police car's two sources, the fire engine's
three, the car's with its horn) in its Movement box.

- **The road vehicles** are the example's sources of their names: point sources 0.05 m
  above the road, where CNOSSOS-EU puts a vehicle's sound, each with a walk along its
  lane with Doppler, left out of the example's own simulations until ticked (there the
  traffic stays a still line source). A walk goes through points at a speed, or through
  keyframes timed for one steady speed, since CNOSSOS-EU gives a vehicle's sound at a
  steady speed: keyframes that speed up or slow down by more than 1 % are refused,
  unless `--speed` drives the vehicle through their places at one speed. The demo's
  scene (`scene_demo.yaml`) has the vehicle alone along its walk at that speed, with
  CNOSSOS-EU's sound power of its category at that speed on the method's reference road
  surface, in the scene's air temperature: rolling noise A_R + B_R lg(v/70) + K (20 - t)
  and propulsion noise A_P + B_P (v - 70)/70 per octave band, with the coefficients of
  Table F-1 of Directive 2002/49/EC, Annex II, as amended by Commission Delegated
  Directive (EU) 2021/1226 (`acoustic_spaces.demos.road`). At their speeds and 20 °C:

  | vehicle | CNOSSOS-EU category | speed | sound power | A-weighted |
  |---|---|---:|---:|---:|
  | `car` | 1, light motor vehicles | 50 km/h | 102.0 dB | 98.4 dB(A) |
  | `bus` | 2, medium heavy vehicles (two axles) | 50 km/h | 109.4 dB | 104.5 dB(A) |
  | `truck` | 3, heavy vehicles (three axles or more) | 70 km/h | 114.1 dB | 110.2 dB(A) |
  | `motorcycle` | 4b, motorcycles (propulsion noise alone) | 50 km/h | 104.1 dB | 97.5 dB(A) |

  The amendment has the vehicle's point radiate alike every way, where Directive (EU)
  2015/996 had it radiate into the half space above the road, and its coefficients A
  are higher: the car's sound power is 2.1 to 2.7 dB(A) higher than by 2015/996's table
  from 30 to 130 km/h, 2.5 dB(A) at 50 km/h (96.0 dB(A) before acoustic-spaces 4.1.0;
  `validation/urban_sources.py`). The table is used as published: no measured pass-by
  is compared with it here. Measured pass-bys of medium heavy vehicles in France found
  the 2015 table's levels 3 to 6 dB(A) low (Euronoise 2018, paper 210).
- **Their sound** is made to that power. A recording of a vehicle going past would not
  do: it holds its own Doppler shift, the road and the distance, and the simulation
  would add them again. Tyre noise is noise with the rolling noise's octave spectrum;
  the engine is a pulse each time one of its cylinders fires (every 15 ms for the car's
  four at 2000 rpm, 17 ms for the truck's six at 1200, 13 ms for the bus's six at 1500
  and 15 ms for the motorcycle's two at 4000, the cylinders a little unequal and the
  firing a little uneven) with the propulsion noise's spectrum; the two are at the
  ratio of their sound powers in each band, so the octave levels are CNOSSOS-EU's and
  the time structure is made up, as the stills say. The motorcycle has the engine
  alone, as the method gives two-wheelers no rolling noise. The sound fades in and out
  over its first and last second, where the vehicle is farthest. `--input
  car=engine.wav` plays a dry recording instead (on board, or at the wheel), mixed down
  to mono and at least as long as the pass; the captions still give CNOSSOS-EU's
  vehicle.
- **The police car** (in the street) is two sources along one walk: the car itself
  (`police_car`), a passenger car 0.05 m above the road with CNOSSOS-EU's sound power
  and a sound made as the car's (its engine four cylinders at 2000 rpm, `--rpm`), and
  its siren (`police_car_siren`), the walk's points at the height of its roof, 1.5 m up,
  facing the way it drives (its walk's `face_forward`). Each is simulated along the walk
  alone, and the two are heard together: their sounds added, each at its own sound
  power, and the energies of their levels band by band. The demo refuses a siren whose
  walk goes elsewhere in plan or at other times than the car's. `--police-car siren` plays the siren alone. The siren
  has the level and the directivity of a commercial 100 W electronic siren the National Bureau of Standards
  measured in an anechoic room (NBS Special Publication 480-28, 1978): 3 m away, its
  yelp averaged over 20 s, 118 dB(A) on its axis, 112 dB(A) 50 degrees off it and 107
  dB(A) at 90 degrees. The scene's `siren.csv` goes through those three, in dB linearly
  with the angle off the axis, alike round the axis and behind as at 90 degrees: NBS
  measured neither the vertical nor behind, so both are assumptions
  (`acoustic_spaces.demos.siren`). Its sound is a square wave at the siren's
  fundamental, as the siren's Class D amplifier drives its loudspeaker: the wail and
  the yelp sweep it from 500 to 1500 Hz and back 12 and 180 times a minute (California's
  rule for sirens asks 10 to 30 and 150 to 250), the two tones are 440 and 586.7 Hz, a
  fourth apart, 1.5 s each (the Swiss rule for priority vehicles asks base frequencies
  of 360 to 630 Hz in the ratio 3:4 and a cycle of 2.5 to 3.5 s). The loudspeaker is
  taken to send every harmonic alike and the same power in every pattern, the yelp's:
  130.9 dB in all, 130.6 dB(A) for the wail and the yelp and 128.6 dB(A) for the two
  tones, which sit lower where the ear weighs less. At 50 km/h the car's sound power is
  32 dB(A) under the wail's and 30 under the two tones', but a square wave sends its odd
  harmonics alone, nothing under its fundamental: in the octaves of 63 to 250 Hz the
  police car's sound is the car's, the rumble of it going by (the siren's `level_db`
  gives those octaves 44 to 55 dB under its whole power).
  Against the rules (`validation/urban_sources.py`): 3.0 m away the wail and the yelp
  give 118.0 dB(A) on the axis and 112.0 dB(A) 50 degrees off it, above the least
  levels of California's class B siren at every angle of its table and 1 to 2 dB under
  those of class A (the rule takes the average of five peaks of the sweep, NBS the
  energy over 20 s), loudest in the 1 kHz octave band, as the rule asks (1 or 2 kHz),
  and the wail's level in 125 ms falls 2.2 dB(A) within its cycle, where the rule
  allows 10. The two tones give 119.4 dB(A) 2 m away on the axis in a free field (the
  Swiss rule's test bench: 116 to 129 dB(A)) and 108.1 to 110.8 dB(A) 7 m in front of
  the car over the street's asphalt, 0.5 to 1.5 m up (on the vehicle: 100 to 115
  dB(A)).
- **The fire engine** (in the street) is three sources along one walk: the vehicle
  itself (`fire_engine`), a medium heavy vehicle with two axles (CNOSSOS-EU's category
  2, as the bus is) 0.05 m above the road, with a sound made as the bus's (six cylinders
  at 1500 rpm, `--rpm`); its siren (`fire_engine_siren`) on its cab's roof, 3.0 m up;
  and its air horn (`fire_engine_horn`), 1.2 m up. The siren and the horn face the way
  it drives, with the directivity of the siren NBS measured (`siren.csv`), and the three
  are heard together, each at its own sound power (`--fire-engine` leaves out the horn,
  the vehicle itself, or both). The siren plays the police car's two tones (440 and
  586.7 Hz, 1.5 s each), 6.0 dB louder: as loud as the Swiss rule lets two tones be on a
  vehicle, 114.9 dB(A) 7 m in front of it at the loudest of 0.5 to 1.5 m up, over the
  street's asphalt (the rule: 100 to 115 dB(A)), 134.6 dB(A) of sound power. The horn is
  as loud as the rule for vehicles' horns (Council Directive 70/388/EEC, Annex I) lets
  one be on its test bench: 118 dB(A) 2 m in front of it on its axis, in a free field
  (the rule: 105 to 118). The rest of the horn is assumed, as no measured air horn of a
  fire engine was found (`acoustic_spaces.demos.horn`): two trumpets sounding together
  at 150 and 180 Hz, a minor third apart as a car's pair of horns is (about 405 to 420
  Hz and 500 Hz), within the 125 to 180 Hz Wikipedia's article on vehicle horns gives
  trucks' air horns, without a source; each a sawtooth (all its harmonics, each 1/k of
  the fundamental); two blasts of 0.6 s, 0.3 s apart, every 3 s, each at its level 0.05
  s after it starts (the rule: within 0.2 s); the height of the rule's bench; and the
  siren's directivity. Its sound power in the scene is its average over its blasts and
  pauses, 122.7 dB(A), as its made sound has them: during a blast it is 4.5 dB louder
  (127.1 dB(A)) than that. The captions' levels have the horn as loud as its sound plays
  around each place: its level there is the average's times how loud its sound is over
  the time around that place (from halfway to the place before to halfway to the next)
  against its average, faded in and out with it. A recording given for it (`--input
  fire_engine_horn=WAV`) plays its blasts as loud as the made horn's, whatever its own
  pauses, and the captions follow its own blasts. Against the rules
  (`validation/urban_sources.py`): the siren gives 125.4 dB(A) on the Swiss rule's bench
  (116 to 129) and 112.3 to 114.9 dB(A) on the vehicle; the horn 107.9 to 108.6 dB(A) 7
  m in front of the vehicle, 0.5 to 1.5 m up (the rule: 93 at least), and, on its bench,
  110.2 dB(A) between 1800 and 3550 Hz (the rule: 105 at least), its loudest component
  there 6.0 dB over any above 3550 Hz (the rule: over).
- **The car's horn** (in the street, `--car car_and_horn`, or `horn` alone) is a pair of
  disc horns behind the car's front bumper (`car_horn`), 0.5 m up along the car's walk,
  facing the way it drives, with the siren's directivity, in two blasts of 0.25 s, 0.15
  s apart, every 3 s from the start of the pass, each at its level 0.02 s after it
  starts (the blasts, the height and the directivity assumed). The horns are Hella's
  S77, approved to UN Regulation 28: 360 and 430 Hz, 110 +/- 3 dB(A) 2 m in front of
  them (Hella's sheet for its S70, S77 and S98), within the rule's 105 to 118. A disc
  horn's diaphragm strikes its magnet's core every cycle and its tone disc rings, its
  main components near 3 kHz, harmonics of the diaphragm's frequency (US patent
  8,689,723): each horn's harmonics are struck alike and shaped by a resonance at 3 kHz
  with a Q of 2 (assumed), which leaves the two fundamentals 24 and 23 dB under it. Its
  sound power is 119.1 dB(A) during a blast and 110.9 dB(A) averaged over its blasts and
  pauses, as the scene has it; the captions follow its blasts as the fire engine's
  horn's do. Against the rule (`validation/urban_sources.py`): 109.9 dB(A) on its bench,
  107.6 dB(A) between 1800 and 3550 Hz (105 at least), its loudest component there 1.9
  dB over any above 3550 Hz, and 100.4 to 101.3 dB(A) 7 m in front of the car, 0.5 to
  1.5 m up (93 at least). Made as the air horn is, a sawtooth at each fundamental, it
  would give only 102.8 dB(A) between 1800 and 3550 Hz, under the rule. `--car car`, the
  default, plays the car alone, as it plays without a horn.
- **The pass**: the vehicle every metre along its walk (`--step`), simulated at each
  listener in turn (`acoustic-spaces walkthrough`), its sound played through those
  places with each early path through its own delay as it changes ([moving sources and
  Doppler](spatial-audio.md#moving-sources-and-doppler)), the late sound crossfaded from
  place to place without a shift of its own. Through places 1 m apart a car's sound at 1
  kHz follows the exact field's waveform to within -54 dB (-57 dB a quarter of a metre
  apart; [validation](validation.md#moving-sources-against-the-exact-field)). The level
  at each moment, the loudest (L_Amax) and the sound exposure level (SEL) come from the
  same simulation, each place's as if the vehicle stood there (a horn's as loud as it
  sounds around that place); the stills give them, with how close the vehicle comes and
  how far the pitch of its direct sound falls from where it starts to where it ends. One
  gain makes every pass, and each is turned up to the loudness of the loudest, by as
  much as the still says.
- **The roadside**: 150 m of road, the car in the lane on the meadow's side at 50 km/h,
  the truck in the lane on the yard's side the other way at 70 km/h, heard at the verge,
  23 and 48 m away over the grass and 27 m away over the paved yard (from the car as it
  passes; the truck passes 4 m farther from the meadow's listeners and nearer the
  yard's). The car's pass peaks at 76.3 dB(A) at the verge, 5.7 m away (SEL 76.2
  dB(A)), 54.3 and 43.5 dB(A) over the grass and 63.2 dB(A) over the yard; the truck's
  at 84.4 dB(A) at the verge, 9.6 m away (SEL 85.3 dB(A)), 67.5 and 56.5 dB(A) over the
  grass and 76.7 dB(A) over the yard, 23.0 m away. Over the grass the ground takes away
  much of the middle and the treble, the more the farther away, while over the yard its
  reflection adds to the direct sound. At the verge the engine's tone falls 1.4
  semitones as the car passes and 1.9 as the truck does, as the geometry says.
- **The street**: 125 m of street, the car, the motorcycle and the fire engine in the
  lane on the offices' side, the bus and the police car in the lane on the houses' side
  the other way, the tram down the middle of the street (see [trains passing
  by](#trains-passing-by)), all at 50 km/h, heard on the pavement (6.2 m from the car,
  the motorcycle and the fire engine's siren as they pass, 2.5 m from the bus and 2.0 m
  from the police car's siren), on a balcony of the office block and in the garden
  behind the noise barrier. The passes peak at (L_Amax, dB(A)):

  | | car | bus | motorcycle | police car (wail) | fire engine | tram |
  |---|---:|---:|---:|---:|---:|---:|
  | pavement | 79.4 | 90.3 | 78.1 | 114.8 | 113.0 | 95.6 |
  | balcony | 74.0 | 80.3 | 73.4 | 104.5 | 111.1 | 91.1 |
  | garden | 62.7 | 71.3 | 61.5 | 95.1 | 97.2 | 73.7 |

  The police car's column is its siren's, which the car itself does not change at this
  precision: the car goes where the bus does, with a sound power at least 4.6 dB under
  the bus's in every octave band, so its passes peak at least 28 dB(A) under the siren's
  at every listener and add 0.01 dB at most. It is heard under the siren's pitch, which
  dB(A) weighs little. The fire engine's column is its three sources together, its siren
  much the loudest: in the garden the siren alone peaks at 97.2 dB(A), the horn's blasts
  at 90.0 and the truck at 68.4, and the fire engine peaks there as its siren does,
  between two of the horn's blasts; on the pavement a blast comes with the siren's peak
  and adds 0.7 dB to it (113.0 dB(A), the siren alone 112.3). The balcony hears the fire
  engine nearly as loud as the pavement does, as its siren, 3.0 m up in the lane on the
  offices' side, passes 7.6 m from the balcony and 6.2 m from the pavement. With its
  horn (`--car car_and_horn`) the car peaks at 97.1 dB(A) on the pavement, 94.0 on the
  balcony and 81.6 in the garden, its horn's loudest blast 17.6 to 19.9 dB over the
  car's own loudest.

  The garden hears the car loudest where the car sees it straight through the 4 m gap
  between the houses and the barrier (14 m short of the gap), less while the barrier
  stands between them, and more again once the car comes out past the barrier's far
  end. The police car, coming the other way, the garden hears loudest 16 m before the
  barrier's far end, before the barrier stands between them, the garden some 20 degrees
  off the siren's axis; the pavement hears it loudest 2 m before it passes, 45 degrees
  off the axis, rather than as it passes, 90 degrees off.
  Where a reflection through the gap comes or goes, it fades in or out over a few
  metres, as in a real field: every path, also the copy of a reflection off the road
  (which the car, so close to it, makes nearly as strong and in step with it) and the
  second-order reflections, fades at the edges that cut it off. From one metre of road
  to the next the garden's early sound from the car changes by at most 3.1 dB(A),
  smoothly where paths go in and out of step; 1 cm apart, the steps left stand out by
  0.55 dB(A) at most (see [Validation](validation.md#along-a-street)). The late sound comes from rays
  drawn at random, few of which reach the garden, so two runs of the car's pass put its
  level up to 2.1 dB(A) apart at one place (its L_Amax within 0.01 dB, the balcony's
  0.17 dB apart) unless the scene has a `seed` (see
  [Scenes](scenes.md#air-response-length-and-seed)); the levels above are one run's
  of each vehicle.
  Reflections off the road, the facades and the barrier and the sound diffracted round
  their edges each shift in pitch their own way. Between the buildings the late sound,
  which keeps the engine's pitch, is strong in the bass, so the engine's tone falls less
  plainly than by the open road.

## Trains passing by

`passby_demo.py` comes with the `railway` example too, and runs a train past its
listeners; in the street it runs a tram down the middle of the street (`--vehicle
tram`). Each listener hears one pass in turn, its pitch falling as it goes by, and the
video follows the train on the plan, with the level heard over the pass drawn under it.

```sh
acoustic-spaces examples --copy . railway && cd railway
python passby_demo.py all --train freight   # or sound, scene, walk, video one at a time
```

Options: `--train regional` the train (`regional`, an electric locomotive and five
coaches, 151 m long, at 160 km/h; `s_bahn`, a four-car electric multiple unit, 67 m, at
120 km/h; `intercity`, the locomotive and twelve coaches, 336 m, at 200 km/h;
`freight`, the locomotive and 24 wagons with composite brake blocks, 446 m, at 100
km/h), `--speed` how fast it runs (km/h; by default the train's own above, whatever
the speed of the example's walk: the Run demo window's **Speed** reads *the train's
own* at zero), `--track ballast` the track it runs on (`ballast`, sleepers in ballast;
`slab`, slab track; `slab_absorber`, slab track with absorbers; `level_crossing`),
`--horn off` the train's warning horn (`blasts` or `chord` sounds it, see below), `--indoors off`
where it is heard (see below), `--step 1` how
far apart the train is simulated along its walk (m), `--hrtf me.sofa` a measured head,
`--input train=WAV` a dry recording of yours (`train_roof=WAV` and
`train_pantograph=WAV` for the sources up on the train, `train_horn=WAV` for its horn).
The tram has no options of its own but `--indoors`: a low-floor tram with air
conditioning, 30 m long, on its track set in the street, at the speed of its walk (50
km/h) unless `--speed` says.

- **Their sound power** is that of Schall 03 (Germany's method for the noise of
  railways and tramways, Anlage 2 of the 16. BImSchV, 2014; `acoustic_spaces.demos.rail`):
  for each vehicle unit, from its data sheet (Beiblatt 1 for trains, 2 for trams), each
  of its sources' A-weighted sound power per metre of track for one unit an hour, per
  octave band, at its speed and on its track (equation 1), with the speed factors of
  Tabelle 6 and 14 and the track's corrections of Tabelle 7 and 15. A unit going by at
  v km/h once an hour has the sound power of that level plus 10 lg(1000 v) dB. The
  sources sit at the rail top (rolling noise, the bogies' aerodynamic noise, the fans
  under the floor, the motors), on the roof, 4 m up (its aerodynamic noise and its
  fans), and at the pantograph, 5 m up, each radiating louder across the track than
  along it, by 10 lg(0.22 + 1.27 sin² δ) dB with δ the angle to the track (equation 8;
  the example's `train.csv`, the street's `tram.csv`). At their speeds on ballast (the
  tram on its street track):

  | train | units | length | speed | sound power |
  |---|---|---:|---:|---:|
  | `regional` | locomotive (Fz 7) + 5 coaches (Fz 9) | 151 m | 160 km/h | 130.6 dB(A) |
  | `s_bahn` | electric multiple unit (Fz 5) | 67 m | 120 km/h | 121.7 dB(A) |
  | `intercity` | locomotive + 12 coaches | 336 m | 200 km/h | 136.8 dB(A) |
  | `freight` | locomotive + 24 wagons (Fz 10) | 446 m | 100 km/h | 131.8 dB(A) |
  | `tram` | low-floor tram (Fz 21) | 30 m | 50 km/h | 115.0 dB(A) |

  The tram's sound power is 16.6 dB(A) over the car's at 50 km/h (CNOSSOS-EU's, 98.4
  dB(A)): Schall 03's tram, with air conditioning on its roof (+8 dB on that source),
  on a track set in the street, whose rolling noise Tabelle 15 makes 1 to 8 dB louder
  per octave band than on ballast.
- **A train as long as it is.** Scene sources walk as points, so the example has three:
  `train` at the rail top, `train_roof` and `train_pantograph`, each with a walk along
  the track (the street's `tram` and `tram_roof`). The demo shares each unit's sources
  at a height out between points along it: at the rail top its bogies, from 15 to 85 %
  of its length (a coach's two bogies 18.5 m apart, as a UIC coach's pivots are), on the
  roof one over each car, and the pantograph in the locomotive's middle. Each source
  plays the train's front point at its height along its walk, and every other point
  there along the same walk as far behind as it is: the same places, that much later, so
  the regional train passes for 151 m. Each point has its own sound (noise with the
  octave spectrum of its sources there) at its own sound power, each heard through the
  same simulated places with its own Doppler, and the levels add each point's energy
  that much later. The units' lengths and the points' places are assumptions; Schall 03
  gives only each unit's sound power per metre of track. Its sound's time structure
  (steady noise) is made up too; its spectrum per octave band is Schall 03's.
- **The horn** (`--horn blasts`, the example's `train_horn`, at the train's front 2.0 m
  above the rail top, along its walk): its high note (660 Hz) and its low note (370 Hz)
  in turn, the notes of the EU's rule for locomotives and passenger trains (TSI
  LOC&PAS, point 4.2.7.2.1), a second each, 0.2 s apart, every 4 s, each as loud as EN
  15153-2 lets it be, 109 dB(C) 25 m ahead of the train over an open, flat site
  (`acoustic_spaces.demos.horn.TRAIN_HORN`). Each note is an air horn's sawtooth, rich in
  harmonics, its sound power taken from that level with the site's ground reflection
  taken out, harmonic by harmonic: 140.2 dB(A) while the high note sounds and 139.2
  while the low one does, 136.4 dB(A) averaged over its blasts and pauses, 5.8 dB(A)
  over the regional train's. `--horn chord` sounds the two notes together in the same
  blasts, which EN 15153-2 allows, the two 109 dB(C) together (its level for a group of
  horns sounded at once; `TRAIN_CHORD`): 139.7 dB(A) while it sounds, 136.4 averaged.
  Its blasts, its height on the train, the measuring site's ground and microphone
  height, and its directivity (the measured siren's) are assumptions; see
  [Validation](validation.md#sirens-and-road-vehicles-against-their-rules).
- **Indoors** (`--indoors`; `acoustic_spaces.demos.indoors`): a listener 0.5 to 2 m in
  front of a building's wall, where ISO 1996-2 measures in front of a façade, hears the
  pass in the room behind it, and the others hear it outdoors. The railway's
  `house_18m` stands 2 m in front of a house 20 m from the track; the street's pavement
  and balcony stand in front of its houses and its offices. `train_and_rumble` plays the
  train through the room's closed window and the rumble the ground's vibration makes in
  it, `train` the train alone, `rumble` the rumble alone (`tram_and_rumble` and `tram`
  for the tram). The room is EN ISO 12354-3's, not the building's geometry: AGC's
  4-16-4 double glazing as measured by EN ISO 10140, Rw (C; Ctr) = 30 (-1; -4) dB, 1.82
  m² of it into a room of 50 m³. Each octave band in the room is the simulated level at
  the listener, the façade's reflection in it, less D_2m,nT = R + 10 lg(V / (6 T0 S)),
  the window's sound reduction index plus 9.6 dB (`acoustic_spaces.facade`); for
  traffic's spectrum that is Rw + Ctr + 9.6 = 34.6 dB. The rumble is the FTA's general method (see [Vibration
  through the ground](outdoor.md#vibration-through-the-ground)) at each moment of the
  pass: the train's curve at the distance from its nearest wheels to the room's wall
  behind the listener, the floor 3 dB over the ground, its rooms' rumble 50 dB under that
  in dB(A), played as noise in the third octave at 25 Hz, the same in both ears: in the
  railway's house 27.5 dB(A) for the S-Bahn and 41.3 for the regional
  train. The room's own reverberation is not added, and the sound keeps the direction it
  comes from outdoors.
- **What is felt**: each listener's caption gives how hard the floor of its house, or
  outdoors the ground under it, shakes at most during the pass, in VdB (the FTA's
  curves, which are the ground's), and whether that is felt, from about 65 VdB (the
  manual's threshold of perception). Outdoors nothing rumbles: the manual counts
  ground-borne noise only inside buildings. Each listener's vibration is also a track
  for a bass shaker, `walk/<listener>.shaker.wav`, and the tour's `tour.shaker.wav`:
  noise in the third octave at 25 Hz whose RMS is the vibration's velocity, every
  listener at one gain, its full scale in `walk.json` (`shaker`).
- **Against Schall 03's own sum** (`validation/rail_sources.py`): in a free field, the
  engine's pass of the S-Bahn at 80 km/h and of the regional train at 160 km/h, 7.5 m
  from the track 1.2 m above the rail top and 25 m away 3.5 m above it, gives the same
  SEL as Schall 03 summing the same stretch of track in pieces of 0.1 m, within 0.01 dB
  in every octave band. Over the railway example's ground the engine gives 2.9 to 4.0 dB(A) less than Schall 03
  with its own terms for the ground, which reckon the ground's reflection otherwise (see
  [Validation](validation.md#trains-against-schall-03-and-the-eus-limits), which also
  sets Schall 03's units beside the EU's limits for new trains).
- **The railway**: 300 m of track, each train at its own speed on ballast, heard 7.5 m from the
  track 1.2 m above the rail top (`tsi_7m5`), 25 m away over the meadow (`meadow_25m`)
  and 25 m away on the other side, behind the barrier (`behind_barrier_25m`). The
  passes peak at (L_Amax, with the SEL after it, dB(A)):

  | | regional, 160 km/h | S-Bahn, 120 km/h | intercity, 200 km/h | freight, 100 km/h |
  |---|---:|---:|---:|---:|
  | 7.5 m | 95.3 (99.5) | 89.6 (91.8) | 98.2 (104.7) | 91.3 (102.7) |
  | meadow, 25 m | 87.7 (93.2) | 82.1 (85.4) | 90.7 (98.4) | 84.2 (96.2) |
  | behind the barrier, 25 m | 78.2 (79.1) | 66.2 (69.6) | 82.3 (83.2) | 74.7 (79.3) |

  The longer a train, the more its SEL is over its L_Amax: the freight train, 446 m
  long, takes 16 s to pass at 100 km/h, the S-Bahn, 67 m, 2 s at 120 km/h. The barrier,
  3 m high and 4 m from the track, hides the rail top from the listener behind it, but
  not the roof (4 m up) or the pantograph (5 m up): behind it the regional train peaks
  9.5 dB(A) under its level on the meadow's side, the S-Bahn, whose rail top is 20 dB
  over its roof in sound power (the regional train's 16 dB), 15.9 dB(A) under.
- **The tram in the street**: the tram runs down the middle of the street at 50 km/h, 4.0 m from the
  pavement's listener, and peaks at 95.6 dB(A) on the pavement (SEL 99.3), 91.1 on the
  balcony (95.4) and 73.7 in the garden (80.3): louder than the bus, which passes 2.5 m
  from the pavement's listener at 90.3 dB(A), as Schall 03's tram, on its track set in
  the street, has 10.5 dB(A) more sound power than CNOSSOS-EU's bus at 50 km/h.

## A helicopter flying over

`flyover_demo.py` comes with the outdoor example `helicopter_flyover`: an Airbus
Helicopters AS350 B3 flies over a meadow 150 m up at 63 m/s, against a light wind, and
each listener hears one pass in turn: right under its track, 1.2 m above the grass,
where the microphone of its noise certification stands, and 150 m to its left and to
its right. It is the [road vehicles' demo](#vehicles-passing-by) with the helicopter in
the car's place (`acoustic_spaces.demos.helicopter`): the same steps, options and video,
but for the car's sound.

```sh
acoustic-spaces examples --copy . helicopter_flyover && cd helicopter_flyover
python flyover_demo.py all   # or helicopter, scene, walk, video one at a time
```

Options: `--speed 200` how fast it flies over the ground (km/h; without it, the speed of
its walk, 226.8 km/h), `--broadband bpm` its rotors' broadband noise from each blade
section's own noise instead of Pegg's method ([Helicopters](outdoor.md#helicopters)),
`--step 4` how far apart it is simulated along its walk (m), `--hrtf me.sofa` a
measured head, `--input helicopter=WAV` a dry recording of yours, `--listeners
under_track` the listeners it is simulated and heard at. Its 900 m of flight
take 14.3 s at 63 m/s.

- **The helicopter** is the example's source `helicopter`, a point at the main rotor's
  hub, with a walk 150 m up with Doppler, facing the way it flies. Its rotors are
  trimmed for level flight ([Helicopters](outdoor.md#helicopters)), so a walk that
  climbs or descends by more than 0.5 m is refused. They are trimmed at its airspeed:
  its speed over the ground less the wind of the scene's own weather at its height, as
  vectors, in the scene's air. In the example the wind blows against it at 4.5 m/s
  150 m up (3 m/s at 10 m), so it flies through the air at 67.5 m/s. It faces along
  its walk, so it does not turn into a wind from the side, as a helicopter flying
  crabwise would: a wind along its track, or none, suits it best. The demo's scene
  (`scene_demo.yaml`) has the helicopter alone with the sound power per octave band and
  the directivity of its rotors' sound at that airspeed, the directivity in a table
  beside it (`scene_demo_helicopter.csv`): 138.9 dB, 133.1 dB(A) in the example, which
  its own scene has too.
- **Its sound** is what the rotors send each way: each harmonic of each rotor's blade
  passing frequency with a pattern and a phase of its own, and broadband noise. Each
  early path plays what is sent the way it leaves the helicopter, at the time it leaves
  ([the sound sent each way](spatial-audio.md#moving-sources-and-doppler)); the late
  sound plays it averaged over the directions. It fades in and out over its first and
  last second. `demo/helicopter/below.wav` is what they send straight down, dry, to
  listen to. `--input helicopter=WAV` plays a dry recording instead, mixed down to mono,
  at least as long as the pass, sent every way with the rotors' directivity; the
  captions still give the rotors' model.
- **The pass**: the helicopter every 4 m along its walk, simulated at each listener in
  turn, as the car's. From 150 m up, places 4 m apart follow the exact field of a
  flyover to within 55 dB at 4 kHz, more closely than the car's places 1 m apart
  ([validation](validation.md#moving-sources-against-the-exact-field)). Over the grass
  the bands up to 500 Hz come from the wide-angle parabolic equation where the sound
  comes in at 34° from the horizontal or flatter (`parabolic_equation: wide_angle`, as
  the narrow-angle one holds within some 15°), and the rays give the rest
  ([how the simulation bends sound](outdoor.md#how-the-simulation-bends-sound)). The
  wind bends the sound the helicopter sends ahead up and the sound it leaves behind
  down, and the eddies (`moderate` at 2 m, weaker higher up) make it flicker in the
  bands the rays give.

What it leaves out: the rotors' model is louder than the helicopter's noise
certificate. Flown over the microphone as the certification does, it gives a sound
exposure level of 88.0 dB(A) where the AS350 B3 is certified at 84.6 dB(A), nearly all
of it the main rotor's broadband noise by Pegg's method; with `--broadband bpm` it gives
80.1 dB(A), its tail rotor's tones then nearly all of it (see
[Validation](validation.md#helicopter-rotors-against-theory-and-a-certified-level)).
The helicopter keeps one trim along its walk, as in steady level flight: the slap of
the blades meeting the vortices of those ahead, as a helicopter makes descending or
turning, and the loads that change round each turn are left out. The eddies flicker
at the pace of their drift in the wind, not that of the helicopter sweeping the paths
through them, and they are checked for sources up to 10 m up, not 150 m.


## A helicopter over the stadium concert

`helicopter_demo.py`, the stadium example's second demo, flies the
[flyover's helicopter](#a-helicopter-flying-over) over the
[stadium concert](#a-concert-in-the-stadium) halfway through the song, and adds its pass to each
listener's mix of the concert, both at their simulated levels. The helicopter is in the
stadium's scene, left out of the concert's own simulations: the editor shows its walk,
the Run demo window lists it as what the demo moves, and what Edit movement… changes
there the demo flies. It builds on the stadium
demo: run that one first, in the same work folder, up to its `render` step, with the
`--format` and the `--hrtf` this one gets.

```sh
acoustic-spaces examples --copy . stadium_concert && cd stadium_concert
python stadium_demo.py all        # the concert first (see above)
python helicopter_demo.py all     # or flight, helicopter, scene, walk, mix, video one at a time
```

Options: `--height 330` how high it flies (m, at every place of its walk; by default
as the scene has it), `--at 180` when it is over the centre
spot, in seconds into the song as the sound system plays it (by default halfway
through); `--speed`, `--broadband`, `--step` and `--input helicopter=WAV` as in the
flyover demo, and `--tour` and `--listeners` as in the stadium demo (the helicopter is
then simulated at those listeners alone, over the concert's mixes there). At the mix position the walk takes 3.3 minutes on 4
cores, most of it playing the 301 places through with Doppler, so some 40 for the 12
listeners.

- **The flight.** The stadium's scene has the helicopter, left out of the concert's own
  simulations, flying 1200 m along the stadium's long axis at 63 m/s, from behind the
  stage over the centre spot to past the east stand, with its rotors' sound power and
  directivity (`helicopter.csv`) at that speed through still air. `flight` writes
  `scene_flight.yaml`: the stadium's scene with the helicopter at `--height` when that
  is given. As the scene has it, it flies 150 m up, as a police or a television helicopter
  given leave may. Without leave the EU's rules of the air keep a flight by sight (VFR)
  over an open-air assembly of persons 300 m above the highest obstacle within 600 m
  (SERA.5005(f)), here the stadium's roof, 30 m up: `--height 330`. The ground and the
  sky are widened to take the flight, and the responses are as much longer as the
  helicopter is farther than the stage (4.1 s against the concert's 2.0 s), so each
  keeps the stadium's decay after the helicopter's sound arrives.
- **The helicopter and its pass** are the flyover demo's steps on that scene:
  `helicopter` its rotors' sound, `scene` the helicopter alone (`scene_overhead.yaml`)
  with the sound power and the directivity of its rotors at 63 m/s through still air
  (138.3 dB, 132.9 dB(A)), and `walk` the helicopter every 4 m along its flight at
  each of the 12 listeners, with Doppler. At the mix position it peaks at about
  81 dB(A) (about 74 from 330 m), where the sound system's loudspeakers, simulated at
  their sound power, give 97 dB(A): it is some 15 dB under the music, heard most in the
  song's quiet passages. High in the north stand, under the roof, it peaks 10 dB lower.
- **The mix.** `mix` adds the helicopter's pass at each listener to the concert's mix
  there (`demo/seats`), the helicopter over the centre spot at `--at`. The feeds stand
  for the loudspeakers' sound power, a gated RMS of 0.1 for the loudest one's (the main
  arrays', 134 dB), and the helicopter's sound is in pascals 1 m from its hub, so the
  mix scales it by 0.1 over the pressure 1 m from a source of the main arrays' sound
  power, in the scene's air. The mixes get one gain for all (the loudest peak at
  -1 dBFS), into `demo/crossing/seats`, and the feeds' table goes to
  `demo/crossing/feeds` with the song lasting until the helicopter has gone.
- **The video** is the stadium demo's, made from those mixes into `demo/crossing`: the
  same tour, each stop turned up to the loudness of the loudest, the helicopter named
  in the title and in the credit. The tour goes on as the stadium demo's does: halfway
  through each of the three songs it is in the north stand, at its far seats and then
  at its top row, under the roof, where the helicopter is quietest. `--at 10` brings
  the helicopter over during the first stop, at the mix position.

What it leaves out: the stadium's air is still, so no wind or eddies bend the
helicopter's sound. The rotors' downwash does not reach the crowd: flying past at this
speed, their wake trails behind them and sinks a few metres a second (1.6 m/s through
the main rotor's disc in its trim). The rotors' model is 3.4 dB louder than the
helicopter's noise certificate, as in the flyover. The stills draw the stadium as the
stadium demo's do, without the helicopter, and the concert's mixes are taken as they
are: one rendered with other HRTFs than `--hrtf` names is not caught.

## Road works in the street

`works_demo.py` comes with the `street` example too: road works in the lane on the
houses' side, 20 m along from the pavement's listener, heard on the pavement, on the
balcony and in the garden in turn, binaurally, and a video that tours them.

```sh
acoustic-spaces examples --copy . street && cd street
python works_demo.py all   # or scene, feeds, simulate, render, video one at a time
```

Options: `--blows 480` how many blows a minute the breaker strikes (Epiroc's MB 1650, a
1650 kg breaker, strikes 320 to 640), `--seconds 20` how long each listener is heard,
`--listeners pavement,garden` which listeners and in which order (every one in use by
default; `--tour`, its earlier name, still works), `--indoors off` where they are heard
(see below), `--hrtf me.sofa` a measured head, `--input
NAME=WAV` a dry recording of yours instead of a made sound (`breaker`, `generator`,
`dumper` or `dumper_alarm`). The scene with the works alone goes to `scene_works.yaml`,
everything else to `demo_works/`. In the editor, **Simulation > Run demo…** on the
street's scene has it beside the pass-by demo.

- **The machines** are rows of Defra's noise database for construction sites (*Update
  of noise database for prediction of noise on construction and open sites*, Hepworth
  Acoustics for Defra, 2005, the data BS 5228-1 Annex C gives;
  `acoustic_spaces.demos.plant`): each one's octave-band levels 10 m away, its sound
  power 28 dB over them, a point over hard ground radiating into a half sphere.

  | source | Defra's row | height | 10 m away | sound power |
  |---|---|---:|---:|---:|
  | `breaker` | a 1650 kg breaker on a 15 t excavator of 121 kW, breaking up brick foundations (table 1, row 9) | 1.0 m | 90 dB(A) | 118.3 dB(A) |
  | `generator` | a 6 kVA diesel generator at 3000 rpm (table 4, row 87) | 0.5 m | 65 dB(A) | 92.9 dB(A) |
  | `dumper` | a 9 t dumper of 75 kW, the loudest moment of its drive-by (table 4, row 4) | 1.5 m | 76 dB(A) | 103.5 dB(A) |

  The heights are assumed. Their sound is made to that power: the breaker's blows, a
  burst of noise each at `--blows`, and the engines' firing, the generator's at its
  3000 rpm (one cylinder, assumed) and the dumper's at 2200 rpm (four cylinders,
  assumed), each brought to its row's octave spectrum. Defra's levels are averages
  over an activity, its pauses too; the made sounds are steady.
- **The reversing alarm** (`dumper_alarm`), at the dumper's back, 1.2 m up, facing the
  works: a tonal alarm at 1200 Hz, 107 dB(A) 1.2 m in front of it on its axis (SAE J994
  Type B; Hella's 009148021 is one), on and off for as long, 1.5 times a second (J994
  asks for 1 to 2; the rate is assumed within that), as directional as the siren NBS
  measured (assumed, as for the fire engine's air horn). Its sound power is 111.7 dB(A)
  while it sounds and 108.6 dB(A) on average over its beeps, which the scene holds. The
  pavement's listener is in front of the dumper, behind its alarm, 11 dB under its axis.
- **What each place hears** (the simulated level of each source, dB(A)):

  | | breaker | generator | dumper | alarm | all |
  |---|---:|---:|---:|---:|---:|
  | pavement | 91.0 | 65.4 | 77.8 | 79.7 | 91.5 |
  | balcony | 89.0 | 62.9 | 75.7 | 77.3 | 89.5 |
  | garden | 72.1 | 48.0 | 57.7 | 58.5 | 72.5 |

  The houses stand between the works and the garden, which hears them over and round
  the houses and the barrier, 19 dB(A) under the pavement.
- **A tone is heard as the street makes it at its frequency.** The table's levels are
  each octave band's, as a sound spread over the octave has them. The alarm's beeps are
  one tone, where the reflections off the road and the facades add up or cancel: they
  come out 2.9 dB under the table's level on the pavement, 0.7 dB over it on the
  balcony and 5.3 dB under it in the garden (the simulated responses at 1200 Hz, which
  change a few tens of centimetres away). The engines' firing, a row of tones, comes
  out within 1.1 dB of the table's; the breaker's blows, noise, within 0.4 dB. The
  captions give the octaves' levels.
- **Indoors** (`--indoors`; `acoustic_spaces.demos.indoors.StillHouses`): as in [Trains
  passing by](#trains-passing-by), the pavement's listener, 2 m in front of the houses,
  and the balcony's, 1 m in front of the offices, hear the works in a room behind that
  wall, and the garden hears them outdoors: `works_and_rumble` the works through the
  closed window and the rumble of the breaker's and the dumper's vibration, `works` the
  window alone, `rumble` the rumble alone. The vibration is each machine's level at its
  distance from the room's wall by the FTA's eq. 7-3 from table 7-4 (a hoe ram, 87 VdB
  at 25 ft; loaded trucks, 86 VdB), their energies added, with no adjustment for the
  building, as the manual compares plant's level with its limits as it is; the rumble is
  50 dB under it in dB(A), as the trains'. It is 29.8 dB(A) in the houses (their floor
  at 79.8 VdB) and 25.5 in the offices (75.5), and steady, as the breaker's blows come
  faster than noise at 25 Hz can swell and fade. The garden's ground shakes at 66.1
  VdB, just felt. The captions give what is felt at each listener, and each one's
  vibration is a bass shaker's track (`<listener>.shaker.wav` beside the mixes,
  `tour.shaker.wav`), as the trains' are.

What it leaves out: the dumper stands still while its alarm sounds, as if it reversed on
the spot; the machines' bodies are not in the scene, so nothing of the dumper shields
the alarm, and Defra's levels are taken as coming from one point at each machine's
height. See [Validation](validation.md#road-works-against-defras-rows-and-the-eus-limits)
for the rows against their own levels and the EU's limits for outdoor machines.

## Aircraft taking off and landing

`flight_demo.py` comes with the outdoor example `airport`: Doc 29's hypothetical
wing-engined twin jet JETW (ECAC Doc 29 Volume 3) takes off from a runway across a
meadow, or lands on it, and each listener hears its pass in turn, binaurally: under the
departure 4 km from its start of roll, beside the runway 450 m from its centre line
(2.5 km from the start of roll) and under the approach 2 km before the threshold, all
1.2 m above the grass. It is the [road vehicles' demo](#vehicles-passing-by) with the
jet in the car's place (`acoustic_spaces.demos.aircraft`): the same steps (the first
one `aircraft`), options and video, but no `--speed`, since Doc 29 times the flight.

```sh
acoustic-spaces examples --copy . airport && cd airport
python flight_demo.py all                     # the departure
python flight_demo.py all --vehicle arrival   # the arrival
```

Options: `--vehicle arrival` the landing instead of the take-off, `--step 4` how far
apart the jet is simulated along its flight (m), `--hrtf me.sofa` a measured head,
`--input departure=WAV` a dry recording of yours, `--listeners lateral_450m` the
listeners it is simulated and heard at.

- **The flight.** The example's sources `departure` and `arrival` are points whose
  walks give the ground track, along the runway's centre line in the example. The demo
  flies JETW along it as Doc 29's Appendix B does, at Volume 3's default weights in the
  scene's air (15 °C at sea level, ANP's reference headwind of 8 kt), and writes the
  flight into the demo's scene as the walk's keyframes, its pitch falling as it passes
  (Doppler). The departure flies the ICAO-A noise abatement procedure: from standing at
  its start of roll, it lifts off 1493 m down the runway at 301 km/h, climbs at
  take-off thrust to 1500 ft (457 m, 4.0 km from the start of roll), cuts back to
  climb thrust by 4.3 km and is 740 m up at the end of its walk, 6.5 km out, after 95.0
  s. The arrival comes down a 3° approach from 225 m up at 253 km/h, 120 m up over the
  approach's listener, touches down 290 m past the threshold, puts on reverse thrust
  and rolls out to 50 km/h 1.6 km down the runway, after 92.7 s. A walk of your own
  gives another ground track (Edit movement…); the departure's starts where its walk
  does, and the arrival touches down where its walk first comes down to the runway, at
  most 1 m up.
- **Its sound** is noise with the octave spectrum of its sound power, which comes from
  its NPD curves and spectral class ([an aircraft in a
  scene](outdoor.md#aircraft-by-ecac-doc-29)): 155.7 dB(A) at the departure's highest
  thrust (25,000 lb, at its start of roll) and 145.4 dB(A) at the arrival's (its reverse
  thrust, 10,000 lb), louder and softer as the thrust changes along the flight: the
  departure's climb thrust 8.0 dB under its start of roll, the arrival's approach 2.3
  dB under its reverse thrust and the end of its roll-out 3.5 dB under. Its directivity is its engines' installation
  correction (under the wings), in a table beside the demo's scene
  (`scene_demo_departure.csv`). `demo/departure/engines.wav` is the made sound, dry, as
  long as the flight. `--input departure=WAV` plays a dry recording instead, at least as
  long as the flight, its level following the recording's.
- **The pass**: the jet every 4 m along its flight at each listener in turn, through
  the direct sound and its reflection off the ground; the late sound is crossfaded from
  place to place. Places 4 m apart follow the exact field of a flyover closely from 100
  m away and more ([validation](validation.md#moving-sources-against-the-exact-field)).
  A flight of 6.5 km is some 1600 places at each listener, which takes a while.

While the jet is in the air, what each listener hears is within 0.9 dB of Doc 29's own
sum over the same flight path's segments, but for the arrival's SEL at
approach_2000m, 1.7 dB under. While it rolls on the runway they part by up to 11.4 dB,
as the engine reckons the sound running along the ground in still air over the scene's
runway and meadow, where Doc 29 takes off an attenuation it has from measurements
([the check](validation.md#an-aircraft-in-the-engine-against-doc-29)).

What it leaves out: one aircraft, Volume 3's hypothetical JETW, and no houses. Its
sound is noise with one spectrum all along, with none of a jet's tones (its fan's, its
buzz-saw noise), and the NPD curves give no directivity along the flight path but
behind the take-off roll, so the source has none either. The captions' L_Amax is the
highest level of the jet's places, each as heard with the jet standing there, with no
time weighting; Doc 29's is measured with slow time weighting, which would lower a
point source's loudest moment 200 ft away at 160 kt by 1.8 dB, 0.2 dB at 1000 ft.
