# Example scenes

| Example | What it is |
| --- | --- |
| `shoebox` | An 8 × 6 × 3 m room from an OBJ file, the simplest case every method must get right. |
| `lecture_hall` | A fan-shaped lecture hall built from its floor plan, no CAD file needed. |
| `concert_hall` | A 32 × 20 × 15 m shoebox hall with a raised stage and upholstered seats; two sources on stage, three seats in the stalls. |
| `studio` | A 6 × 4.5 × 3 m control room with side panels, a diffuser and stereo loudspeakers at ±30°. |
| `church` | A stone nave with an apse, 18 m high, with about six seconds of reverberation. |
| `gallery` | An L-shaped gallery: one listener sees the guide, the other stands round the corner and hears the guide by [edge diffraction](simulation.md#edge-diffraction). |
| `gallery_walk` | The gallery with a visitor walking away from the guide and round the corner while the guide talks, for a [walkthrough](spatial-audio.md#walkthroughs) with its video. |
| `archway` | A bar and a dining room joined by an archway in a wall 40 cm thick: a diner behind the wall hears the singer over both corners of the arch's jamb (`second_order` diffraction). |
| `listening_room` | A living room with a loudspeaker against the front wall or a metre out: the wall's reflection adds to the bass in pressure, or takes from it ([early reflections in pressure](simulation.md#low-frequencies-early-reflections-in-pressure)); `bass.py` prints how much. |
| `leaky_classroom` | A classroom model with two mistakes that leave it open, a missing window pane and a ceiling 3 cm short of a wall, for the [leak finder](scenes.md#finding-leaks) to find. |
| `material_libraries` | A seminar room whose materials come from a CATT-Acoustic file and an ODEON library ([material libraries](scenes.md#material-libraries-of-catt-acoustic-and-odeon)). |
| `bim_office` | An office from a BIM model in IFC, read as one room from its space boundaries, its surfaces named by element, material and IFC class ([BIM models](scenes.md#bim-models-ifc); the `ifc` extra). |
| `rehearsal_room` | A band's rehearsal room under a pitched roof from a Collada file in inches, as SketchUp exports it. |
| `orchestra` | The concert hall with a full orchestra on its stage, one source for each of the 61 tracks of an anechoic recording, and the scripts that [mix them and make a video](spatial-audio.md#an-orchestra-in-the-concert-hall). |
| `town_hall_pa` | A stone town hall of 4000 m³ with a sound system for speech: two columns (coherent [arrays](scenes.md#sources-and-directivity) steered down onto the seats) beside the stage, and two delay loudspeakers further down the hall that play later, quieter and with less bass. `delays.py` maps the STI with and without the delays: they help the back rows and cost the front a little. |
| `workshop` | Noise in a workshop hall: a conveyor 20 m long (a line source), a vibrating machine casing (a surface source) and a compressor, each with its own sound power spectrum. Map the A-weighted level over the floor, then with another ceiling. |
| `apartment` | Two rooms of a flat, a television in one and a bed in the other: how loud the television is through the [wall and the door](scenes.md#walls-doors-and-openings), closed or open. |
| `room_modes` | The bass in a small bare room from the [wave solver](simulation.md#low-frequencies-the-wave-solver), in third-octave bands, on the CPU or a GPU, against rays alone. |
| `fitting` | A seminar room that measured longer than it was designed, as the panels on its back wall were left out: `make_measurements.py` makes up the measurements, then [compare](measurements.md#comparing-with-measurements) the model with them and [fit](measurements.md#fitting-materials-to-measurements) the back wall to them. |
| `street` | Outdoors: traffic along a street between a row of houses and an office block, heard on the pavement, on a balcony and in a garden behind a noise barrier, where the sound comes over the barrier and round its ends ([outdoor scenes](outdoor.md)). |
| `roadside` | Outdoors: traffic on a road between a meadow and a paved yard, heard over the grass at 7.5, 25 and 50 m and over the yard at 25 m: the [ground dip](outdoor.md#ground-impedance) over the grass at a few hundred hertz, which the hard yard does not have. |
| `open_air_stage` | Outdoors: a stage heard 100 and 300 m away across a park in a 5 m/s wind, downwind and upwind: the sound [bent](outdoor.md#wind-and-temperature) down to the ground downwind, sooner and louder in the low bands, and up over the listeners' heads upwind, into a shadow 300 m away that the wind's [eddies](outdoor.md#turbulence) scatter sound into. |
| `tree_belt` | Outdoors: traffic on a road heard from two gardens 55 m away, one behind a belt of trees 30 m deep and one across an open meadow: the trees' [foliage and trunks](outdoor.md#vegetation) take up to about 8 dB from the highest bands and almost nothing from the lowest, and scatter some of the sound into the late part. |
| `measurement` | No scene, a script: it records a sweep in a simulated room, as a microphone would, to try the [measurement](measurements.md#measuring-impulse-responses) commands without a sound card. |

`acoustic-spaces examples` lists them with where they are installed, and
`acoustic-spaces examples --copy my-rooms` copies them to a folder you can edit. From
Python, `acoustic_spaces.examples.example_path("church")` gives a scene file.
