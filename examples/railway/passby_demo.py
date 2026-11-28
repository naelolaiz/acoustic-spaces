"""A train running past on the railway across the meadow, heard binaurally where the EU's
rule for the noise of trains measures one going by (7.5 m from the track's centre, 1.2 m
above the rail top), 25 m away over the meadow, 2 m in front of the wall of a house 20 m
from the track (`house_18m`) and 25 m away behind the noise barrier, its pitch falling
as it goes by (Doppler); and a video that follows.

This is the demo of the railway's scene (``acoustic-spaces demo scene.yaml --sources
train,train_roof,train_pantograph``, This scene in the editor): a preset of it, which
keeps the command lines of the script it was. The train is the scene's three sources
``train`` (at the rail top), ``train_roof`` (4 m up) and ``train_pantograph`` (5 m up),
left out of its simulations until picked, at the heights Schall 03 (Germany's method for
the noise of railways, 2014) gives a train's sources, each of a Schall 03 model: a
regional train (an electric locomotive and five coaches) at 160 km/h on sleepers in
ballast. ``--train`` picks another: an S-Bahn (a four-car electric multiple unit, 120
km/h), an intercity train (an electric locomotive and twelve coaches, 200 km/h) or a
freight train (an electric locomotive and 24 wagons with composite brake blocks, 5 of
them tank wagons, 100 km/h), at its own speed unless ``--speed`` says, and ``--track``
another track (slab track, with absorbers, a level crossing); several, comma separated,
play one after the other in one video (``--train regional,freight``). Each unit's
sources have Schall 03's sound power per octave band for that speed and track, shared
out between its points: its bogies at the rail top, its cars' roofs, its pantograph;
each source plays the train's points at its height one behind the other, as far behind
its front as they are, so the train goes by as long as it is (the regional train 151
m), each with its own sound, noise with the octave spectrum of its sources there;
``--input NAME=WAV`` plays a dry recording of yours at each of them instead.
``--sources`` picks which of the train's sources play (all three by default);
``--sources Train`` (the group of the four) sounds the train's warning horn too, the
scene's ``train_horn`` at its front 2.0 m above the rail top: its high note (660 Hz) and
its low note (370 Hz) in turn, each 109 dB(C) 25 m ahead, the most EN 15153-2 allows, a
second each every 4 s; ``--horn chord`` sounds the two together. ``--indoors
airborne_and_rumble`` has ``house_18m`` hear the train inside the house, through its
closed window (EN ISO 12354-3), with the rumble the ground's vibration makes in its room
(the FTA's general method; ``airborne`` through the window alone, ``rumble`` the rumble
alone; the script's ``train_and_rumble`` and ``train`` read as the first two); the other
listeners hear it outdoors. Run it from the folder of the railway example
(``acoustic-spaces examples --copy . railway && cd railway``), in the Python environment
acoustic-spaces is installed in, with ffmpeg on the PATH::

    python passby_demo.py all      # every step, leaving out what is there already
    python passby_demo.py feeds    # the train's sound, a point a column, made to its power
    python passby_demo.py render   # the train every metre along the track, at every listener
    python passby_demo.py video    # demo/passby_binaural.mp4 (an animation), its stills

The script's steps run as the ones that took them over: ``sound`` the feeds, ``scene``
the simulation of the scene with the train taken in, ``walk`` the mixes; ``--from``
names the scene. Each run goes to a folder in ./demo named after what it changes
(demo/passby_train-freight_binaural/), with the video, its audio and its thumbnail beside
it, so that a run with other settings writes over none of them. Use headphones;
``--hrtf me.sofa`` plays it through your own or measured HRTFs instead of a spherical
head.

Each listener hears each of the train's points through every early path the scene
traces (the reflection off the ground, the barrier's top and ends diffracted), each
through its own delay as it changes, so each shifts in pitch as it passes; the late
sound is crossfaded from place to place. The train's points appear one after the other
where its trajectory starts and leave where it ends, each faded in and out. The stills
give the level heard at each moment, the loudest (L_Amax) and the sound exposure level
(SEL) of the pass, as simulated: every point's, each as far behind as it goes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

#: The train's sources, which the script played unless --sources said.
TRAIN = ("train", "train_roof", "train_pantograph")
#: What the script's ``--indoors`` took, as the demo's takes it: the train through the
#: window is the airborne sound.
INDOORS = {"train_and_rumble": "airborne_and_rumble", "train": "airborne"}


def old_indoors(value: str) -> str:
    """A value of the script's ``--indoors`` as the demo's takes it (:data:`INDOORS`)."""
    return INDOORS.get(value, value)


DEMO = preset(
    name="passby",
    example="railway",
    title=N_("A train running past"),
    description=__doc__,
    script=Path(__file__),
    defaults={"sources": ",".join(TRAIN)},
    renamed={"scene": (("--from", str),), "indoors": (("--indoors", old_indoors),)},
    steps={"feeds": ("sound",), "simulate": ("scene",), "render": ("walk",)},
)

__all__ = ["DEMO", "INDOORS", "TRAIN", "old_indoors"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
