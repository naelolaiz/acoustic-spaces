"""A train running past on the railway across the meadow, heard binaurally where the EU's
rule for the noise of trains measures one going by (7.5 m from the track's centre, 1.2 m
above the rail top), 25 m away over the meadow and 25 m away behind the noise barrier,
its pitch falling as it goes by (Doppler); and a video that follows.

The train is the example's three sources `train` (at the rail top), `train_roof` (4 m
up) and `train_pantograph` (5 m up), left out of its simulations until ticked, at the
heights Schall 03 (Germany's method for the noise of railways, 2014) gives a train's
sources; --train picks it: a regional train (an electric locomotive and five coaches,
160 km/h), an S-Bahn (a four-car electric multiple unit, 120 km/h), an intercity train
(an electric locomotive and twelve coaches, 200 km/h) or a freight train (an electric
locomotive and 24 wagons with composite brake blocks, 100 km/h), at its own speed unless
--speed says, on sleepers in ballast unless --track says (slab track, with absorbers, a
level crossing). Each unit's sources have Schall 03's sound power per octave band for
that speed and track, shared out between its points: its bogies at the rail top, its
cars' roofs, its pantograph; each source plays the train's points at its height one
behind the other, as far behind its front as they are, so the train goes by as long as
it is (the regional train 151 m), each with its own sound, noise with the octave
spectrum of its sources there; `--input NAME=WAV` plays a dry recording of yours at
each of them instead. Run it from the folder of the railway example (`acoustic-spaces
examples --copy . railway && cd railway`), in the Python environment acoustic-spaces is
installed in, with ffmpeg on the PATH:

    python passby_demo.py sound   # demo/train/train.wav, train_roof.wav, ...: dry, a column a point
    python passby_demo.py scene   # scene_demo.yaml: the train alone, at --speed, with Doppler
    python passby_demo.py walk    # the train every metre along the track, at every listener
    python passby_demo.py video   # demo/tour.wav, demo/tour.mp4 (an animation), stills

or `python passby_demo.py all --train freight`, every step, leaving out what is there
already; the editor's Simulation > Run demo… runs it too. Use headphones; `--hrtf
me.sofa` plays it through your own or measured HRTFs instead of a spherical head.

Each listener hears each of the train's points through every early path the scene
traces (the reflection off the ground, the barrier's top and ends diffracted), each
through its own delay as it changes, so each shifts in pitch as it passes; the late
sound is crossfaded from place to place. The train's points appear one after the other
where its walk starts and leave where it ends, each faded in and out over a second. The
captions give the level heard at each moment, the loudest (L_Amax) and the sound
exposure level (SEL) of the pass, as simulated: every point's, each as far behind as it
goes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos import passby
from acoustic_spaces.demos.rail import TRAIN_VEHICLE

DEMO = passby.demo(Path(__file__), "railway", "A train running past", __doc__, (TRAIN_VEHICLE,))

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
