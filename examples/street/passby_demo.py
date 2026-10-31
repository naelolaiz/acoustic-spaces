"""A car, a bus, a motorcycle or a police car with its siren on driving along the street
between the houses and the office block, heard binaurally on the pavement, on a balcony
and in the garden behind the noise barrier, its pitch falling as it goes by (Doppler);
and a video that follows.

The vehicles are the example's own (the sources `car`, `bus`, `motorcycle` and
`police_car`, left out of its simulations until ticked), at 50 km/h (--speed); --vehicle
picks the one that goes past. The car, the bus and the motorcycle send their sound from
0.05 m above the road with CNOSSOS-EU's sound power per octave band for that speed and
their category (Directive 2002/49/EC, Annex II, Appendix F, Table F-1, as amended in
2021): a passenger car, a city bus with two axles, a motorcycle. Their sound is made to
that power: tyre noise (none for the motorcycle), and the firing of their engine (four
cylinders at 2000 rpm, six at 1500 and two at 4000; --rpm), each with the spectrum of
its part; `--input NAME=WAV` plays a dry recording of yours instead (one of a vehicle
going past has its Doppler and the road in it already). The police car plays its siren
from its roof, 1.5 m up (--siren: wail, yelp or two-tone), at the level and with the
directivity of a 100 W electronic siren the National Bureau of Standards measured. Run
it from the folder of the street example (`acoustic-spaces examples --copy . street &&
cd street`), in the Python environment acoustic-spaces is installed in, with ffmpeg on
the PATH:

    python passby_demo.py sound   # demo/NAME/NAME.wav (police_car/siren.wav): the sound, dry
    python passby_demo.py scene   # scene_demo.yaml: the vehicle alone, at --speed, with Doppler
    python passby_demo.py walk    # the vehicle every metre along the street, at every listener
    python passby_demo.py video   # demo/tour.wav, demo/tour.mp4 (an animation), stills

or `python passby_demo.py all --vehicle bus`, every step, leaving out what is there
already; the editor's Simulation > Run demo… runs it too. Use headphones; `--hrtf
me.sofa` plays it through your own or measured HRTFs instead of a spherical head.

Each listener hears the vehicle through every early path the scene traces (reflections
off the road, the facades and the barrier up to the third order, and the direct sound
and the first reflections diffracted round their edges), each through its own delay as
it changes, so the direct sound and each reflection shift in pitch as the vehicle
passes, each its own way; the late sound between the buildings is crossfaded from place
to place, without a shift of its own. The captions give the level heard at each moment,
the loudest (L_Amax) and the sound exposure level (SEL) of the pass, as simulated; the
vehicle's walk, edited in the editor's Movement panel, is driven as edited, at --speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos import passby
from acoustic_spaces.demos.siren import POLICE_CAR_VEHICLE

DEMO = passby.demo(
    Path(__file__),
    "street",
    "A vehicle driving down the street",
    __doc__,
    (passby.THE_CAR, passby.BUS, passby.MOTORCYCLE, POLICE_CAR_VEHICLE),
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
