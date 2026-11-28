"""Doc 29's twin jet JETW taking off from the aerodrome or landing on it, heard
binaurally under its departure 4 km from its start of roll, beside the runway 450 m from
its centre line and under its approach 2 km before the threshold, its pitch falling as
it passes (Doppler); and a video that follows.

The jet is the example's source `departure` or `arrival` (--sources picks it, or
--vehicle; the arrival is left out of the example's own simulations until ticked), each
a point with a Doc 29 model whose trajectory gives its ground track. The model flies it
along that track as ECAC Doc 29 flies JETW (its Appendix B, at Volume 3's default
weights, in the scene's air): the departure by the ICAO-A noise abatement procedure from
its start of roll, the arrival down a 3° approach to touch down where its trajectory
first comes down to the runway, then its reverse thrust and roll-out. The heights,
speeds and thrust along the way are Doc 29's, and so is how long the flight takes (the
example's departure 95 s, its arrival 93 s), so --speed changes nothing of it. Its sound
power comes from its NPD curves (its maximum level 1000 ft under it at the flight's
highest thrust, with its spectral class's spectrum) and its directivity from its
installation correction (engines under the wings). Its sound is noise with that
spectrum, louder and softer as the thrust changes; `--input departure=WAV` plays a dry
recording of yours instead. This is the demo of the scene (This scene in the editor),
with the script's command lines. Run it from the folder of the airport example
(`acoustic-spaces examples --copy . airport && cd airport`), in the Python environment
acoustic-spaces is installed in, with ffmpeg on the PATH:

    python flight_demo.py aircraft  # the feeds: its engines along the flight, dry
    python flight_demo.py scene     # the simulation of the scene the run plays
    python flight_demo.py walk      # the jet every 4 m along its flight, at every listener
    python flight_demo.py video     # demo/flight_binaural.mp4 (an animation), stills

or `python flight_demo.py all --sources arrival`, every step, leaving out what is there
already; the editor's Simulation > Run demo… runs it too. Each run goes to ./demo, named
after its settings, so that the arrival writes over nothing of the departure's;
`--sources departure,arrival` plays the two one after the other in one video. Use
headphones; `--hrtf me.sofa` plays it through your own or measured HRTFs instead of a
spherical head.

Each listener hears the jet's direct sound and its reflection off the ground, each
through its own delay as it changes, so each shifts in pitch as it passes. The jet's
sound fades in and out over a second where its flight starts and ends. The captions give
the level heard at each moment, the loudest (L_Amax) and the sound exposure level (SEL)
of the pass, as simulated, and both as Doc 29 measures them (L_ASmax through a sound
level meter's slow time weighting, the SEL over the time it is within 10 dB of that);
validation/aircraft_noise.py compares them with Doc 29's own sum over the flight.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

DEMO = preset(
    name="flight",
    example="airport",
    title=N_("A jet taking off and landing"),
    description=__doc__,
    script=Path(__file__),
    # The script's departure, and its jets each in a part of their own.
    defaults={"sources": "departure", "sources_play": "in-turn"},
    renamed={"scene": (("--from", str),), "sources": (("--vehicle", str),)},
    steps={"feeds": ("aircraft",), "simulate": ("scene",), "render": ("walk",)},
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
