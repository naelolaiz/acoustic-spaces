"""A car (its horn too, if wanted), a bus, a motorcycle, a police car with its siren on, a
fire engine with its siren and its air horn or a tram going along the street between the
houses and the office block, heard binaurally on the pavement, on a balcony and in the
garden behind the noise barrier, its pitch falling as it goes by (Doppler); and a video
that follows.

The vehicles are the example's own (the sources `car` with `car_horn`, `bus`,
`motorcycle`, `police_car` with `police_car_siren`, `fire_engine` with
`fire_engine_siren` and `fire_engine_horn`, and `tram` with `tram_roof`, left out of its
simulations until ticked), at 50 km/h (--speed); --vehicle picks the one that goes past.
The car, the bus and the motorcycle send their sound from 0.05 m above the road with
CNOSSOS-EU's sound power per octave band for that speed and their category (Directive
2002/49/EC, Annex II, Appendix F, Table F-1, as amended in 2021): a passenger car, a
city bus with two axles, a motorcycle. Their sound is made to that power: tyre noise
(none for the motorcycle), and the firing of their engine (four cylinders at 2000 rpm,
six at 1500 and two at 4000; --rpm), each with the spectrum of its part; `--input
NAME=WAV` plays a dry recording of yours instead (one of a vehicle going past has its
Doppler and the road in it already). `--car car_and_horn` sounds the car's horn too, in
two short blasts every 3 s: a pair of disc horns at 360 and 430 Hz, their tone discs
ringing near 3 kHz, 0.5 m up, 110 dB(A) 2 m in front of them, as Hella's S77 (approved
to UN Regulation 28), with the siren's directivity; `--car horn` plays the horn alone.
The police car is a passenger car made as the car is, with its siren on its roof, 1.5 m
up (--siren: wail, yelp or two-tone), at the level and with the directivity of a 100 W
electronic siren the National Bureau of Standards measured; --police-car siren plays the
siren alone. The fire engine is a medium heavy vehicle with two axles made as the bus
is, its siren on its cab's roof, 3.0 m up, playing the two tones as loud as the Swiss
rule lets them be on a vehicle, and its air horn 1.2 m up, in blasts as loud as the rule
for vehicles' horns lets one be on its bench; --fire-engine leaves out the horn, the
vehicle itself, or both. The tram is a low-floor tram with air conditioning, 30 m long,
on its track set in the street, its sound power Schall 03's (Germany's method for the
noise of railways and tramways, 2014) at the rail top, shared out between its four
bogies, and on its roof, between its two halves; each of its two sources plays its
points at its height one behind the other, so it goes by as long as it is. Run it from
the folder of the street example
(`acoustic-spaces examples --copy . street && cd street`), in the Python environment
acoustic-spaces is installed in, with ffmpeg on the PATH:

    python passby_demo.py sound   # demo/NAME/NAME.wav (and NAME/siren.wav, horn.wav): dry
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
from acoustic_spaces.demos.rail import TRAM_VEHICLE
from acoustic_spaces.demos.siren import FIRE_ENGINE_VEHICLE, HONKING_CAR, POLICE_CAR_VEHICLE

DEMO = passby.demo(
    Path(__file__),
    "street",
    "A vehicle driving down the street",
    __doc__,
    (
        HONKING_CAR,
        passby.BUS,
        passby.MOTORCYCLE,
        POLICE_CAR_VEHICLE,
        FIRE_ENGINE_VEHICLE,
        TRAM_VEHICLE,
    ),
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
