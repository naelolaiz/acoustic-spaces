"""A car passing on the road across the meadow, heard binaurally at the verge, 25 and 50 m
away over the grass and 25 m away over the paved yard, its pitch falling as it goes by
(Doppler); and a video that follows.

The car is the example's own (the source `car`, left out of its simulations until
ticked): a passenger car at 50 km/h (--speed) in the lane on the meadow's side, whose
sound comes from 0.05 m above the road with CNOSSOS-EU's sound power per octave band for
that speed (Commission Directive (EU) 2015/996, Appendix F, Table F-1). Its sound is made
to that power: tyre noise, and the firing of a four-cylinder engine at 2000 rpm (--rpm),
each with the spectrum of its part; `--input car=WAV` plays a dry recording of yours
instead (one of a car going past has its Doppler and the road in it already). Run it
from the folder of the roadside example (`acoustic-spaces examples --copy . roadside &&
cd roadside`), in the Python environment acoustic-spaces is installed in, with ffmpeg on
the PATH:

    python passby_demo.py car     # demo/car/car.wav: the car's sound, dry
    python passby_demo.py scene   # scene_demo.yaml: the car alone, at --speed, with Doppler
    python passby_demo.py walk    # the car every metre along the road, at every listener
    python passby_demo.py video   # demo/tour.wav, demo/tour.mp4 (an animation), stills

or `python passby_demo.py all`, every step, leaving out what is there already; the
editor's Simulation > Run demo… runs it too. Use headphones; `--hrtf me.sofa` plays it
through your own or measured HRTFs instead of a spherical head.

Each listener hears the car through every early path, the direct sound and its
reflection off the ground, each through its own delay as it changes, so the pitch falls
as the car passes; over the grass the ground takes away much of the middle and the
treble, the more the farther away, while over the yard its reflection adds to the
direct sound. The captions
give the level heard at each moment, the loudest (L_Amax) and the sound exposure level
(SEL) of the pass, as simulated; the car's walk, edited in the editor's Movement panel,
is driven as edited, at --speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos import passby

DEMO = passby.demo(Path(__file__), "roadside", "A car passing the meadow", __doc__)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
