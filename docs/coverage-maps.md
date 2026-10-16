# Coverage maps

A coverage map gives the parameters on a grid over the audience area, the way CATT and
ODEON draw them: strength G and sound pressure level, C50, C80, D50, EDT, T20, T30 and
Ts per octave band, and the A-weighted level and speech transmission index.

```sh
acoustic-spaces map examples/lecture_hall/scene.yaml --plot sti.png --parameter sti
acoustic-spaces map examples/concert_hall/scene.yaml --spacing 0.5 --json map.json \
    --screenshot c80.png --parameter c80 --band 1000
```

![Coverage map of the speech transmission index in the lecture hall example, seen from above](images/coverage.png)

The map covers the surfaces named in `coverage.surfaces` (by default those whose names
contain "audience", else the floor) with square columns `spacing` metres wide, with a
point at the listening `height` above the surface in each (above its top, when a
slab's underside is in the same group); a balcony over the stalls gets a second level
of points. Columns along a slanted wall keep the part over the surface, in every room of
a model that holds several (with the sound that passes between them). The early part is
exact at every point, as at a listener. The late part comes from one ray tracing run per
source in which every ray adds its path length through a slab of air around each point
(a track-length estimate), so a map of hundreds of points costs about as much as a few
listeners: the concert hall example (480 points, two sources, 50,000 rays per band)
takes about a minute on 4 cores. In the shoebox example, full simulations at map points
give G within 0.2 dB of the map and T30 within 5 %. With the statistical late method
each point gets the late part a listener there would, and G agrees within 0.01 dB.

Each source radiates its sound power per band (its `level_db`, `spectrum`, `gain_db`
and `eq_db`; see [Sources and directivity](scenes.md#sources-and-directivity)), which sets the
sound pressure levels. The STI (IEC 60268-16, with the male weighting, or the female
one for a `speech_female` source) takes that as the speech level, with auditory masking
and the reception threshold, and `noise_db` as the background noise. In the map of all
sources together, each source's sound arrives its `delay_ms` later (the `town_hall_pa`
example's `delays.py` maps its sound system with and without its delay loudspeakers).
Line and surface sources add up the sound of their points and arrays radiate their
beams, as they do at a listener (the `workshop` example maps the A-weighted noise of
both). `map` prints the range of every parameter, `--plot` saves the top view,
`--screenshot` the 3D view and `--json` every value of every point. Settings live in
the scene:

```yaml
coverage:
  surfaces: [audience_floor]   # default: names with "audience", else "floor"
  height: 1.2                  # listening height above the surface, m
  spacing: 1.0                 # distance between points, m
  rays: 50000                  # rays per source and band (default: simulation.late.rays)
  noise_db: 35                 # background noise for the STI, dB SPL (or one per band)
```

From Python:

```python
from acoustic_spaces.engine import coverage_map
from acoustic_spaces.viz import plot_map, show_room

result = coverage_map("examples/concert_hall/scene.yaml")
both = result.get()  # all sources together; result.get("soloist") for one
c80 = both.value("c80", band=4)  # one value per point, 1 kHz
plot_map(both, "c80.png", parameter="c80", room=result.room)
show_room(result.room, coverage=both, parameter="sti")
```
