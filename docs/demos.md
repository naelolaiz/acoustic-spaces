# Demos

Five demos run recordings through example scenes, mix them at the scenes' listeners and
make a video of the result, a sixth drives a car past the listeners of two outdoor
examples, a seventh flies a helicopter over those of a third and an eighth flies it over
the stadium's concert. Each shows one space and one kind of sound: a symphony orchestra
on the stage of a shoebox concert hall, a choir and an organ in a reverberant stone
church, a band with its sound system and stage monitors in a stadium, the same band's
song carried across a lake on a summer evening, a guide talking to a visitor who walks
away from them through a gallery and then turns their head, a car passing on a road
across a meadow and along a street, its pitch falling as it goes by, and a helicopter
flying over a meadow and low over the stadium during the song. None of them
stands for what the simulator does elsewhere; the [examples](examples.md) cover far
more, and any scene makes a demo of its own with the recordings its sources are given
([a demo of any scene](#a-demo-of-any-scene)).

The scripts come with the examples: `acoustic-spaces examples --copy . NAME` copies them
with the scene. The recordings do not: they are other people's work under their own
licences, named below with each demo, and the church, stadium and lake scripts download
them from where their authors publish them. The videos need [ffmpeg](https://ffmpeg.org)
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
(the stadium has two: its concert first, then the helicopter over it), each reading the
open scene where it would read the example's (a copy is known by its
folder's name or, saved anywhere, by the scene's name: a renamed scene is a scene of
its own). Pick the demo and the step,
fill in the options (they start at the same defaults the command line uses), and it
runs in the background with its messages in the **Messages** panel, **Cancel** stopping
it. The sources are a table, each with what it plays and the file it plays instead:
**Choose WAV…** gives the selected one a file, **Default** takes it back. The
**Movement** box lists the sources and listeners of the scene that have a walk, each
with how it goes, how far and for how long, ticked when it moves in the run: *This
scene*'s ticks start as **Moves** sets them in the scene and change that run only
(`--moving`, below), **As in the scene** ticks them as the scene does again, and an
example's demo shows what it moves (the walk through the gallery moves its visitor, the
pass-by demos their car; the other example demos move nothing). A walker left out of
the scene's simulations is listed only when a demo moves it: the car, which the pass-by
demos put in the scene they write. **Edit movement…** closes the window and opens the
[Movement panel](editor.md) on the selected walker. The window shows the command line it
is about to run, so a run started there can be repeated in a terminal or a script. A
demo reads the scene from its file, so a scene with changes is saved first.

Every still says how its sound was made, at the bottom: the version, the early
reflections (traced or image sources, their order, diffraction) and the late sound with
its rays, as the simulation ran them, the wave solver's bands when it ran, outdoors in
moving air the bent paths and the parabolic equation's bands, the HRTFs the mixes went
through (or the spherical head model) and the sample rate; and under that the command
that makes the video again, with the options it was given and paths by their names (the
song, the HRTFs).

## A demo of any scene

Any scene makes a demo: each source plays a recording of its own, every listener is
heard in turn, and a video tours them. A source names what it plays in the scene file,
or with **Plays** on its page in the editor (**Choose WAV…**; **Silent** takes it away):

```yaml
sources:
  - name: talker
    position: [2.0, 3.0, 1.5]
    level_db: 78
    spectrum: speech_male
    signal: sounds/talk.wav   # a dry recording, relative to the scene's folder
```

```sh
acoustic-spaces demo scene.yaml                 # every step, into demo/ beside the scene
acoustic-spaces demo scene.yaml feeds           # one step: feeds, simulate, render, video
acoustic-spaces demo scene.yaml --seconds 15 --tour front,back --hrtf me.sofa
acoustic-spaces demo scene.yaml --moving none   # every walker standing where it starts
```

- **feeds**: each source's recording mixed down to mono, at the scene's sample rate and
  brought to the same gated RMS (unweighted), so that how loud each plays
  against the others is its sound power in the scene (`level_db`, with its feed's gain
  and equalizer), as the simulation says, and not how loud the recording happens to
  be. Sources without a `signal` stay silent and are listed as such. `--input
  SOURCE=WAV` plays another file instead of a source's signal, without changing the
  scene, and gives a silent source one.
- **simulate**: `acoustic-spaces simulate` on the scene, its binaural output through the
  HRTFs the scene's binaural output names, `--hrtf` instead, or the spherical head
  model when neither names any. A scene changed since its results is simulated again.
- **render**: every listener in use (or those `--tour` names, in its order) hearing
  every source that plays, with one gain for all, so that the listeners' differences in
  level are heard as they are.
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
  it the line says which source plays which file.

**Movement.** Sources and listeners whose [walk](scenes.md#walks) is on move along it
while the recordings play, all on one clock from their start, as `walkthrough --all`
plays them ([walkthroughs](spatial-audio.md#walkthroughs)). Then **render** simulates
the scene with every walker at each moment of its walk, in one simulation, and mixes
every listener hearing the walkers go by, or walking itself, with one gain for all; the
still simulation stays for what the stills say where nothing moves. The video is a
moving picture: the walks dotted, every walker where it is at that moment, and the
level of the sources that play as simulated at that moment; the arrival time is left
out where something moves, and T30 is the one where the walks start. A stop hears its
listener where the clock has it then, so `--tour` puts a walker first to hear its walk
from the start. `--moving` picks the walks of a run without changing the scene:
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

The scripts share one package, `acoustic_spaces.demos`: the steps and options above,
the recordings and their phrases, the simulate command and its HRTF, the mixes and
their one gain, the crossfaded tour, the stills and the video. A new demo describes its
steps and options and gets the rest, including the editor's window.

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
# put movement1.zip, movement2.zip and movement4.zip here, from
# https://doi.org/10.14279/depositonce-6729.2 (560 to 710 MB each)
acoustic-spaces simulate scene.yaml --out results
python orchestra_demo.py tracks   # demo/tracks: the 61 tracks of each movement
python orchestra_demo.py render   # demo/seats: each stop of the tour, binaural (~1 min each)
python orchestra_demo.py video    # demo/tour.wav, demo/stills, demo/tour.mp4, demo/thumbnail.png
```

Options: `--tour 1:stalls-middle,1:conductor,2:stalls-rear` (the seats of a movement take
equal shares of it), `--zips FOLDER` for the zips, `--movements 1,2` for `tracks`,
`--format` for another output the simulation wrote, `--no-mp4` for the audio and stills
only. `demo/chapters.txt` lists the stops' times for a video description. `all` runs
tracks, render and video, after the simulation.

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
tracks each (zips of 560 to 710 MB), downloaded by hand; `tracks` takes the 61 WAV files
of each out of its zip.

## A choir and an organ in the church

![A frame of the church video: the church from above and from the side, with the choir in front of the altar, the organ in the gallery and the conductor facing the choir, a line from each singer to the conductor, the choir close up, the tour and a spectrogram of the sound](images/church_demo.png)

`church_demo.py` puts a choir of 19 real singers in front of the altar of the `church`
example and the organ in the gallery over the entrance, and plays the round "Dona nobis
pacem" three times: the organ alone on full stops, the choir alone, then the choir with
the organ on quiet stops, slowing into a held last chord and the church's reverberation
(1:26 in all). It mixes the piece at six places and makes a video that tours them: under
the organ at the back, along the south wall, the front pews, the conductor, the back row
of the choir and the middle of the nave.

```sh
acoustic-spaces examples --copy . church && cd church
python church_demo.py music     # 19 takes, their annotations, 27 organ samples (about 100 MB)
python church_demo.py scene     # scene_choir.yaml: the church with the choir and the organ
acoustic-spaces simulate scene_choir.yaml --out results_choir --output binaural   # 10-20 min on 4 cores
python church_demo.py feeds     # demo/feeds: one WAV per singer and per side of the organ
python church_demo.py render    # demo/seats: the whole piece at each listener
python church_demo.py video     # demo/tour.wav, demo/stills, demo/tour.mp4, demo/thumbnail.png
```

Options: `--tour nave_back:0,conductor:25,...` (listener:start in seconds),
`--listeners a,b` for `render`, `--format` for another output the simulation wrote
(`ambix3`, `5.0`, ...; the video takes its first two channels), `--no-mp4` for the audio
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
  requests, so only the files needed (about 36 MB) are downloaded, not the 2.5 GB.
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
python stadium_demo.py music     # the reference mix and 7 stems (WAV, 450 MB)
python stadium_demo.py band      # scene_band.yaml: the scene with the band on the stage
acoustic-spaces simulate scene_band.yaml --out results_band --output binaural   # ~3 min on 4 cores
python stadium_demo.py feeds     # demo/feeds: one WAV per loudspeaker and player
python stadium_demo.py render    # demo/seats: one binaural WAV per listener (~1 min each)
python stadium_demo.py video     # demo/tour.wav, demo/stills, demo/tour.mp4, demo/thumbnail.png
```

Options: `--song francium` or `rubidium` for `music` and `feeds` (with `--work
demo_francium` for a folder of its own), `--tour mix_position:1,singer:0.5,east_stand:1`
(listener:phrases of 8 bars, 0.5 for 4; the last one runs to the end), `--listeners a,b`
for `render`, `--format` for another output the simulation wrote, `--no-mp4` for the
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

- **The band.** The example has loudspeakers only, so `band` adds the players as sources
  on the stage, in a group of their own: a drum kit beside the drummer (116 dB sound
  power, omni) playing the drums, the singer's voice (98 dB, subcardioid, facing the
  audience) playing the vocal, and an amplifier behind the guitarist (112 dB, cardioid,
  tilted up) playing the guitar, each from its own stem. On stage and near it the players
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
range requests, so only those 450 MB are downloaded. The stems start 58 ms before the
reference mix; `feeds` lines them up with it by cross-correlation.

"Francium" (127.75 BPM, 4:54) and "Rubidium" (132 BPM, 3:52) by Software-Entwicklungskit,
from the album *Everything is Free*, <https://github.com/ichbinsoftware/everythingisfree>,
**CC0 1.0**: electronic songs released with their stems. `music --song francium` downloads
the master and the 26 stems (M4A, 800 MB in all; "Rubidium" has 9) from the album's
repository at a fixed commit, decodes them with ffmpeg and sums the stems into the groups
the mixes use: drums, bass, guitar (the synth lead, which the guitarist's amplifier
plays), keys, vocal and backing vocals.

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
(listener:weather:phrases of 8 bars), `--last-phrases` (how long the last stop lasts, 0
to the end of the song; 2 by default), `--format` for another output the simulation
wrote, `--no-mp4` for the audio and stills only. `all` runs music, render and video;
`scenes` and the two simulations come before it.

- **The sound system.** `scenes` replaces the example's single source (120 dB sound
  power) with a system as a lawn concert hangs it: two line arrays of eight boxes
  (`lake_array.yaml`, a generic box's pattern in `box.csv`, both made with the stadium
  example's `make_model.py`), 6.5 m up either side of the stage, 124 dB each, aimed down
  the lawn and, box by box, from 5° down at the last rows, where the lawn meets the water,
  to 36° at the front rows, each added up box by box (`near_field`), playing the song's
  left and right channels; and a block of four cardioid subwoofers (`sub_array.yaml`,
  130 dB) on the ground at the stage front, 2 m nearer the audience and delayed 6 ms so
  that both arrive together there, playing both channels, rolled off above 125 Hz. The
  system is tuned at the audience as a system engineer tunes it (`eq_db`: the arrays' low
  mids up, their top octave up for the air): at 30 m every octave from 250 Hz to 4 kHz is
  within 1 dB of 1 kHz, 8 kHz 1.5 dB down and 63 and 125 Hz 2 dB up. `scenes` also adds
  a listener 140 m behind the stage, upwind, and lets the responses run 3.5 s, as the far
  shore hears the stage 1.6 s late.
- **The weather** is the example's: a breeze from the park out over the lake, 2.5 m/s
  10 m up, and two weather zones, the lake's (air 17 °C at the water, a degree warmer
  every 10 m up, and a 3.5 m/s breeze) and the village's (a 1 m/s breeze among the
  houses). The breeze and the cool air bend the sound down to the water, which gives it
  back almost whole. `scene_still.yaml` is the same scene with no atmosphere.
- **What it gives.** At the bedroom window the evening's weather makes the song 2.3 dB(A)
  louder than still air would, in the garden 4.2 dB; behind the stage, upwind, it is
  0.6 dB quieter, and on the boat 1 dB quieter. (With the sound system the example's
  single source was, radiating every way alike, the window gained 4.8 dB: the arrays send
  less of the sound up, where the weather bends it down to the water.)
- **How loud.** From the audience to the far shore the sound drops by 20 to 25 dB(A),
  too much to hear the far places in a video, so each stop is turned up to the
  audience's loudness and its caption says how much quieter the place really is, in
  dB(A). The loudness is K-weighted, as meters weigh it, not A-weighted: the far places
  and the back of the stage keep little but bass (the array boxes radiate every way below
  250 Hz and the water gives the low end back whole), which the A weighting all but
  leaves out, so matching dB(A) would put their bass far above the audience's. The window
  in still air keeps the gain of the window in the evening's weather, so the 2.3 dB is
  heard.
- **The cuts.** Each place's mix is shifted earlier by the time the loudest sound takes
  to reach it (the peak of the energy of its responses: 0.09 s in the audience, 1.65 s
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
  mix (56 MB) is read out of the release's preview archive by HTTP range requests.
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
for the turn, `--turn` the angles it holds, `--hrtf me.sofa` a measured head.

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
through Delft; `speech` reads it out of the corpus's 148 MB archive.

## A car passing by

`passby_demo.py` comes with two outdoor examples, `roadside` and `street`: a passenger
car drives past at 50 km/h and each listener hears one pass in turn, its pitch falling
as the car goes by. The video follows the car on the plan, with the level heard over the
pass drawn under it.

```sh
acoustic-spaces examples --copy . roadside && cd roadside   # or street
python passby_demo.py all   # or car, scene, walk, video one at a time
```

Options: `--speed 70` how fast the car drives (km/h; without it, the speed of its walk
in the example's scene, 50 km/h in both), `--rpm 2000` how fast its engine turns, for its
made sound, `--step 1` how far apart the car is simulated along its walk (m), `--hrtf
me.sofa` a measured head, `--input car=WAV` a dry recording of yours. The car drives the
length of its walk at that speed, so the pass lasts the length divided by the speed: the
150 m of the roadside take 10.8 s at 50 km/h, 7.7 s at 70 km/h. The **Run demo** window
gives the three under its options as they change.

- **The car** is the example's source `car`: a point source 0.05 m above the road, where
  CNOSSOS-EU puts a vehicle's sound, with a walk along the road with Doppler, left out of
  the example's own simulations until ticked (there the traffic stays a still line
  source). Its walk goes through points at a speed, or through keyframes timed for one
  steady speed, since CNOSSOS-EU gives a car's sound at a steady speed: keyframes that
  speed up or slow down by more than 1 % are refused, unless `--speed` drives the car
  through their places at one speed. The demo's scene (`scene_demo.yaml`) has the car
  alone along its walk at that speed, with CNOSSOS-EU's sound power of a passenger car
  (category 1) at that speed
  on the method's reference road surface, in the scene's air temperature: rolling noise
  A_R + B_R lg(v/70) and propulsion noise A_P + B_P (v - 70)/70 per octave band, with
  the coefficients of Table F-1 of Commission Directive (EU) 2015/996
  (`acoustic_spaces.demos.car`). At 50 km/h and 20 °C that is 99.0 dB, 96.0 dB(A).
- **Its sound** is made to that power. A recording of a car going past would not do: it
  holds its own Doppler shift, the road and the distance, and the simulation would add
  them again. Tyre noise is noise with the rolling noise's octave spectrum; the engine is
  a pulse each time one of its four cylinders fires (every 15 ms at 2000 rpm, the
  cylinders a little unequal and the firing a little uneven) with the propulsion noise's
  spectrum; the two are at the ratio of their sound powers in each band, so the octave
  levels are CNOSSOS-EU's and the time structure is made up, as the stills say. It fades
  in and out over its first and last second, where the car is farthest. `--input
  car=engine.wav` plays a dry recording instead (on board, or at the wheel), mixed down
  to mono and at least as long as the pass; the captions still give CNOSSOS-EU's car.
- **The pass**: the car every metre along its walk (`--step`), simulated at each
  listener in turn (`acoustic-spaces walkthrough`), its sound played through those places
  with each early path through its own delay as it changes
  ([moving sources and Doppler](spatial-audio.md#moving-sources-and-doppler)), the late
  sound crossfaded from place to place without a shift of its own. Through places 1 m
  apart a car's sound at 1 kHz follows the exact field's waveform to within -54 dB
  (-57 dB a quarter of a metre apart;
  [validation](validation.md#moving-sources-against-the-exact-field)). The level at
  each moment, the loudest (L_Amax) and the sound exposure level (SEL) come from the
  same simulation, each place's as if the car stood there; the stills give them, with
  how close the car comes and how far the pitch of its direct sound falls from where it
  starts to where it ends. One gain makes every pass, and each is turned up to the
  loudness of the loudest, by as much as the still says.
- **The roadside**: 150 m of road, the car in the lane on the meadow's side, heard at
  the verge (5.7 m from the car as it passes), 23 and 48 m away over the grass and 27 m
  away over the paved yard. The pass peaks at 73.7 dB(A) at the verge (SEL 73.7 dB(A)),
  51.7 and 40.8 dB(A) over the grass and 60.7 dB(A) over the yard: over the grass the
  ground takes away much of the middle and the treble, the more the farther away, while
  over the yard its reflection adds to the direct sound. At the verge the engine's tone
  falls 1.4 semitones as the car passes, as the geometry says.
- **The street**: 125 m of street, the car in the lane on the offices' side, heard on
  the pavement (6.2 m from the car as it passes), on a balcony of the office block and
  in the garden behind the noise barrier. The pass peaks at 76.9 dB(A) on the pavement
  and 71.7 dB(A) on the balcony. The garden hears it loudest, 60.3 dB(A), where the car
  sees it straight through the 4 m gap between the houses and the barrier (14 m short
  of the gap), less while the barrier stands between them, and more again once the car
  comes out past the barrier's far end.
  Where a reflection through the gap comes or goes, it fades in or out over a few
  metres, as in a real field: every path, also the copy of a reflection off the road
  (which the car, so close to it, makes nearly as strong and in step with it) and the
  second-order reflections, fades at the edges that cut it off. From one metre of road
  to the next the garden's early sound changes by at most 3.2 dB(A), smoothly where
  paths go in and out of step, where it stepped by up to 11.4 dB(A) before 4.0.0; five
  small steps are left, the largest 0.63 dB(A) within a centimetre (see
  [Validation](validation.md#along-a-street)). The late sound comes from rays drawn at
  random, few of which reach the garden, so two runs of the pass put its level up to
  2.3 dB(A) apart at one place (its L_Amax within 0.02 dB) unless the scene has a
  `seed` (see [Scenes](scenes.md#air-response-length-and-seed)).
  Reflections off the road, the facades and the barrier and the sound diffracted round
  their edges each shift in pitch their own way. Between the buildings the late sound,
  which keeps the engine's pitch, is strong in the bass, so the engine's tone falls less
  plainly than by the open road.

## A helicopter flying over

`flyover_demo.py` comes with the outdoor example `helicopter_flyover`: an Airbus
Helicopters AS350 B3 flies over a meadow 150 m up at 63 m/s, against a light wind, and
each listener hears one pass in turn: right under its track, 1.2 m above the grass,
where the microphone of its noise certification stands, and 150 m to its left and to
its right. It is the [car's demo](#a-car-passing-by) with the helicopter in the car's
place (`acoustic_spaces.demos.helicopter`): the same steps, options and video, but for
the car's sound.

```sh
acoustic-spaces examples --copy . helicopter_flyover && cd helicopter_flyover
python flyover_demo.py all   # or helicopter, scene, walk, video one at a time
```

Options: `--speed 200` how fast it flies over the ground (km/h; without it, the speed of
its walk, 226.8 km/h), `--step 4` how far apart it is simulated along its walk (m),
`--hrtf me.sofa` a measured head, `--input helicopter=WAV` a dry recording of yours. Its
900 m of flight take 14.3 s at 63 m/s.

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
of it the main rotor's broadband noise (see
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
listener's mix of the concert, both at their simulated levels. It builds on the stadium
demo: run that one first, in the same work folder, up to its `render` step, with the
`--format` and the `--hrtf` this one gets.

```sh
acoustic-spaces examples --copy . stadium_concert && cd stadium_concert
python stadium_demo.py all        # the concert first (see above)
python helicopter_demo.py all     # or flight, helicopter, scene, walk, mix, video one at a time
```

Options: `--height 330` how high it flies (m), `--at 180` when it is over the centre
spot, in seconds into the song as the sound system plays it (by default halfway
through); `--speed`, `--step` and `--input helicopter=WAV` as in the flyover demo, and
`--tour` as in the stadium demo. At the mix position the walk takes 3.3 minutes on 4
cores, most of it playing the 301 places through with Doppler, so some 40 for the 12
listeners.

- **The flight.** `flight` writes `scene_flight.yaml`: the stadium example's scene with
  the helicopter added, left out of the concert's own simulations, flying 1200 m along
  the stadium's long axis at 63 m/s, from behind the stage over the centre spot to past
  the east stand. By default it flies 150 m up, as a police or a television helicopter
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
