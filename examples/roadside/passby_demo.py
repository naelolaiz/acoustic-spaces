"""A car or a truck passing on the road across the meadow, heard binaurally at the verge,
25 and 50 m away over the grass and 25 m away over the paved yard, its pitch falling as
it goes by (Doppler); and a video that follows.

The vehicles are the example's own (the sources `car` and `truck`, left out of its
simulations until ticked); --vehicle picks the one that goes past. The car is a
passenger car at 50 km/h in the lane on the meadow's side, the truck an articulated one
with three axles or more at 70 km/h the other way (--speed). Each sends its sound from
0.05 m above the road with CNOSSOS-EU's sound power per octave band for its speed and
category (Directive 2002/49/EC, Annex II, Appendix F, Table F-1, as amended in 2021). Its
sound is made to that power: tyre noise, and the firing of its engine (four cylinders at
2000 rpm for the car, six at 1200 for the truck; --rpm), each with the spectrum of its
part; `--input NAME=WAV` plays a dry recording of yours instead (one of a vehicle going
past has its Doppler and the road in it already). Run it from the folder of the roadside
example (`acoustic-spaces examples --copy . roadside && cd roadside`), in the Python
environment acoustic-spaces is installed in, with ffmpeg on the PATH:

    python passby_demo.py sound   # demo/car/car.wav (demo/truck/truck.wav): the sound, dry
    python passby_demo.py scene   # scene_demo.yaml: the vehicle alone, at --speed, with Doppler
    python passby_demo.py walk    # the vehicle every metre along the road, at every listener
    python passby_demo.py video   # demo/tour.wav, demo/tour.mp4 (an animation), stills

or `python passby_demo.py all --vehicle truck`, every step, leaving out what is there
already; the editor's Simulation > Run demo… runs it too. Use headphones; `--hrtf
me.sofa` plays it through your own or measured HRTFs instead of a spherical head.

Each listener hears the vehicle through every early path, the direct sound and its
reflection off the ground, each through its own delay as it changes, so the pitch falls
as it passes; over the grass the ground takes away much of the middle and the treble,
the more the farther away, while over the yard its reflection adds to the direct sound.
The captions give the level heard at each moment, the loudest (L_Amax) and the sound
exposure level (SEL) of the pass, as simulated; the vehicle's walk, edited in the
editor's Movement panel, is driven as edited, at --speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos import passby

DEMO = passby.demo(
    Path(__file__),
    "roadside",
    "A car or a truck passing the meadow",
    __doc__,
    (passby.THE_CAR, passby.TRUCK),
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
