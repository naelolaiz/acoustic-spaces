# Example scenes

| Example | What it is |
| --- | --- |
| `shoebox` | An 8 × 6 × 3 m room from an OBJ file, the simplest case every method must get right. |
| `lecture_hall` | A fan-shaped lecture hall built from its floor plan, no CAD file needed. |
| `concert_hall` | A 32 × 20 × 15 m shoebox hall with a raised stage and upholstered seats; two sources on stage, three seats in the stalls. |
| `studio` | A 6 × 4.5 × 3 m control room with side panels, a diffuser and stereo loudspeakers at ±30°. |
| `church` | A stone nave with an apse, 18 m high, with about six seconds of reverberation. |
| `gallery` | An L-shaped gallery: one listener sees the guide, the other stands round the corner and hears the guide by [edge diffraction](simulation.md#edge-diffraction). |
| `archway` | A bar and a dining room joined by an archway in a wall 40 cm thick: a diner behind the wall hears the singer over both corners of the arch's jamb (`second_order` diffraction). |
| `listening_room` | A living room with a loudspeaker against the front wall or a metre out: the wall's reflection adds to the bass in pressure, or takes from it ([early reflections in pressure](simulation.md#low-frequencies-early-reflections-in-pressure)); `bass.py` prints how much. |
| `leaky_classroom` | A classroom model with two mistakes that leave it open, a missing window pane and a ceiling 3 cm short of a wall, for the [leak finder](scenes.md#finding-leaks) to find. |
| `orchestra` | The concert hall with a full orchestra on its stage, one source for each of the 61 tracks of an anechoic recording, and the scripts that [mix them and make a video](spatial-audio.md#an-orchestra-in-the-concert-hall). |

`acoustic-spaces examples` lists them with where they are installed, and
`acoustic-spaces examples --copy my-rooms` copies them to a folder you can edit. From
Python, `acoustic_spaces.examples.example_path("church")` gives a scene file.
