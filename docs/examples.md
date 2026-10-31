# Example scenes

The examples come in groups, the same ones the editor's File > Open example shows,
each from the simplest.

## Rooms

| Example | What it is |
| --- | --- |
| `shoebox` | An 8 × 6 × 3 m room from an OBJ file, the simplest case every method must get right. |
| `room_modes` | The bass in a small bare room from the [wave solver](simulation.md#low-frequencies-the-wave-solver), in third-octave bands, on the CPU or a GPU, against rays alone. |
| `studio` | A 6 × 4.5 × 3 m control room with side panels, a diffuser and stereo loudspeakers at ±30°. |
| `listening_room` | A living room with a loudspeaker against the front wall or a metre out: the wall's reflection adds to the bass in pressure, or takes from it ([early reflections in pressure](simulation.md#low-frequencies-early-reflections-in-pressure)); `bass.py` prints how much. |
| `apartment` | Two rooms of a flat, a television in one and a bed in the other: how loud the television is through the [wall and the door](scenes.md#walls-doors-and-openings), closed or open. |
| `archway` | A bar and a dining room joined by an archway in a wall 40 cm thick: a diner behind the wall hears the singer over both corners of the arch's jamb (`second_order` diffraction). |
| `workshop` | Noise in a workshop hall: a conveyor 20 m long (a line source), a vibrating machine casing (a surface source) and a compressor, each with its own sound power spectrum. Map the A-weighted level over the floor, then with another ceiling. |

## Halls and churches

| Example | What it is |
| --- | --- |
| `lecture_hall` | A fan-shaped lecture hall built from its floor plan, no CAD file needed. |
| `concert_hall` | A 32 × 20 × 15 m shoebox hall with a raised stage and upholstered seats; two sources on stage, three seats in the stalls. |
| `orchestra` | The concert hall with a full orchestra on its stage, one source for each of the 61 tracks of an anechoic recording, and the scripts that [mix them and make a video](demos.md#an-orchestra-in-the-concert-hall). |
| `church` | A stone nave with an apse, 18 m high, with about six seconds of reverberation; `church_demo.py` plays its [choir and organ](demos.md#a-choir-and-an-organ-in-the-church), left out of the church's own simulations until ticked, as are the places it tours, and makes a video. |
| `gallery` | An L-shaped gallery: one listener sees the guide, one stands round the corner and hears the guide by [edge diffraction](simulation.md#edge-diffraction), and a visitor walks away from the guide and round the corner, for a [walkthrough](spatial-audio.md#walkthroughs); `walk_demo.py` makes its [video](demos.md#a-walk-through-the-gallery), where the visitor then looks round from the listener `standing`, left out of the gallery's own simulations until ticked. With the walk turned off the visitor stands where it starts. |

## Sound systems

| Example | What it is |
| --- | --- |
| `town_hall_pa` | A stone town hall of 4000 m³ with a sound system for speech: two columns (coherent [arrays](scenes.md#sources-and-directivity) steered down onto the seats) beside the stage, and two delay loudspeakers further down the hall that play later, quieter and with less bass. `delays.py` maps the STI with and without the delays: they help the back rows and cost the front a little. |
| `concert_stage` | A concert stage on open ground with its monitors: wedges for the singer, a drum fill and side fills, beside the two main line arrays of 12 boxes in a J ([coherent arrays](scenes.md#sources-and-directivity) of a generic box's balloon, added up box by box near the players), the subwoofers and the front fills. `align.py` shows each player hears their own monitor loudest; `feedback.py` finds the gain before feedback of the singer's microphone, a supercardioid listener in front of the singer's voice, both left out of the scene's simulations until ticked, at the mix position and at the singer's ears, and renders it ringing; `near_field.py` prints how far the arrays' far-field patterns would be off at each listener. |
| `stadium_concert` | A concert in a football stadium, with its sound system as tours hang it and the stage's monitors: the two main line arrays, a block of eight cardioid subwoofers timed to arrive with them at the mix position, out-fills, in-fills for the pitch's centre, front fills, two delay towers on the pitch and four delays under the roof's edge, each arriving 6 to 20 ms after the main arrays at its seats, and on stage `concert_stage`'s wedges, drum fill and side fills. Every array adds up its boxes one by one. Map the level over the pitch and the stands; `align.py` prints when and how loud each loudspeaker arrives at each listener on the pitch, the stands and the stage, and which would be heard as an echo; `near_field.py` prints how far the arrays' far-field patterns would be off; `feedback.py` finds the singer's microphone's gain before feedback. Its sound system plays alone with the Stage and Monitors groups left out. `stadium_demo.py` takes in the [band](demos.md#a-concert-in-the-stadium), left out of the concert's simulations as the singer's microphone is, plays a song through it all and makes a video, and `helicopter_demo.py` then flies the scene's helicopter, left out of the concert's simulations, 150 m over the stadium halfway through the song ([a helicopter over the stadium concert](demos.md#a-helicopter-over-the-stadium-concert)). |

## Outdoors

| Example | What it is |
| --- | --- |
| `street` | Outdoors: traffic along a street between a row of houses and an office block, heard on the pavement, on a balcony and in a garden behind a noise barrier, where the sound comes over the barrier and round its ends ([outdoor scenes](outdoor.md)); `--without noise_barrier` gives the barrier's insertion loss. A car, a bus, a motorcycle and a police car with its siren on, left out until ticked, drive along the street at 50 km/h with Doppler: `passby_demo.py --vehicle` plays one ([vehicles passing by](demos.md#vehicles-passing-by)). |
| `roadside` | Outdoors: traffic on a road between a meadow and a paved yard, heard over the grass at 7.5, 25 and 50 m and over the yard at 25 m: the [ground dip](outdoor.md#ground-impedance) over the grass at a few hundred hertz, which the hard yard does not have. A car at 50 km/h and a truck at 70 km/h, left out until ticked, drive along the road with Doppler: `passby_demo.py --vehicle` plays one ([vehicles passing by](demos.md#vehicles-passing-by)). |
| `tree_belt` | Outdoors: traffic on a road heard from two gardens 55 m away, one behind a belt of trees 30 m deep and one across an open meadow: the trees' [foliage and trunks](outdoor.md#vegetation) take up to about 8 dB from the highest bands and almost nothing from the lowest, and scatter some of the sound into the late part. |
| `hillside` | Outdoors: a works yard at the foot of a hill, heard 160 m away in the hill's shadow, on the slope facing the yard and on the crest, over a [terrain](outdoor.md#terrain) read from a grid of heights, with two houses from a map: the sound bends over the crest to the listener behind it, which takes 10 dB off the lowest band and nearly 40 dB off the highest, and reflects off the mean plane of the slope to the one facing the yard. |
| `open_air_stage` | Outdoors: a stage heard 100 and 300 m away across a park in a 5 m/s wind, downwind and upwind: the sound [bent](outdoor.md#wind-and-temperature) down to the ground downwind, sooner and louder in the low bands, and up over the listeners' heads upwind, into a shadow 300 m away that the wind's [eddies](outdoor.md#turbulence) scatter sound into. |
| `lakeside_concert` | Outdoors: a summer evening concert on a lawn by a lake, heard by the audience, from a boat and in a village 550 m away across the water, with two [weather zones](outdoor.md#weather-zones): over the lake the air lies cool and the breeze blows stronger, both bending the sound down to the water, and among the village's houses the breeze drops. The editor's **View > Show wind** and **Show wind profile** draw the three weathers; `lakeside_demo.py` plays a [song across the lake](demos.md#a-concert-across-a-lake) through the stage's sound system, left out of the scene's own simulations until ticked, in that weather and in still air, and makes a video. |
| `helicopter_flyover` | Outdoors: a helicopter, an AS350 B3, flying over a meadow 150 m up at 63 m/s, heard right under its track, where its noise certification's microphone stands, and 150 m to its left and right. Its two rotors' sound comes from [their model](outdoor.md#helicopters), tones and broadband noise, the tones louder to its left, where the main rotor's blades advance; a light wind against it bends the sound it sends ahead up and the sound behind it down, and eddies make it flicker. `flyover_demo.py` plays it past each listener with Doppler ([a helicopter flying over](demos.md#a-helicopter-flying-over)). |

## Models from CAD and BIM files

| Example | What it is |
| --- | --- |
| `rehearsal_room` | A band's rehearsal room under a pitched roof from a Collada file in inches, as SketchUp exports it. |
| `bim_office` | An office from a BIM model in IFC, read as one room from its space boundaries, its surfaces named by element, material and IFC class ([BIM models](scenes.md#bim-models-ifc); the `ifc` extra). |
| `leaky_classroom` | A classroom model with two mistakes that leave it open, a missing window pane and a ceiling 3 cm short of a wall, for the [leak finder](scenes.md#finding-leaks) to find. |

## Measurements and material libraries

| Example | What it is |
| --- | --- |
| `fitting` | A seminar room that measured longer than it was designed, as the panels on its back wall were left out: `make_measurements.py` makes up the measurements, then [compare](measurements.md#comparing-with-measurements) the model with them and [fit](measurements.md#fitting-materials-to-measurements) the back wall to them. |
| `material_libraries` | A seminar room whose materials come from a CATT-Acoustic file and an ODEON library ([material libraries](scenes.md#material-libraries-of-catt-acoustic-and-odeon)). |
| `measurement` | No scene, a script: it records a sweep in a simulated room, as a microphone would, to try the [measurement](measurements.md#measuring-impulse-responses) commands without a sound card. |

`acoustic-spaces examples` lists them with where they are installed, and
`acoustic-spaces examples --copy my-rooms` copies them to a folder you can edit. From
Python, `acoustic_spaces.examples.example_path("church")` gives a scene file.
