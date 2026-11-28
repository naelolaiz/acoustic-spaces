"""A car or a truck passing on the road across the meadow, heard binaurally at the verge,
25 and 50 m away over the grass and 25 m away over the paved yard, its pitch falling as
it goes by (Doppler); and a video that follows.

This is the demo of the roadside's scene (``acoustic-spaces demo scene.yaml --sources
car``, This scene in the editor): a preset of it, which keeps the command lines of the
script it was. The vehicles are the scene's own sources ``car`` and ``truck``, left out
of its simulations until picked; ``--vehicle`` (or ``--sources``) picks the one that
goes past, ``--vehicle car,truck`` each in a part of its own, one after the other in one
video. The car is a passenger car at 50 km/h in the lane on the meadow's side, the truck
an articulated one with three axles or more at 70 km/h the other way (``--speed`` sets
another). Each has a model in the scene: CNOSSOS-EU's sound power per octave band for
its category at its speed, in the scene's air (Directive 2002/49/EC, Annex II, Appendix
F, Table F-1, as amended in 2021), from 0.05 m above the road; its sound is made to that
power: tyre noise, and the firing of its engine (four cylinders at 2000 rpm for the car,
six at 1200 for the truck; ``--rpm``), each with the spectrum of its part.
``--input car=WAV`` plays a dry recording of yours instead (one of a vehicle going past
has its Doppler and the road in it already). Run it from the folder of the roadside
example (``acoustic-spaces examples --copy . roadside && cd roadside``), in the Python
environment acoustic-spaces is installed in, with ffmpeg on the PATH::

    python passby_demo.py all      # every step, leaving out what is there already
    python passby_demo.py feeds    # the vehicle's sound, made to its sound power
    python passby_demo.py render   # the pass every metre along the road, at every listener
    python passby_demo.py video    # the video, its audio and its stills

The script's steps run as the ones that took them over: ``sound`` the feeds, ``scene``
the simulation of the scene with the vehicle taken in, ``walk`` the mixes. Each listener
hears the vehicle through every early path, the direct sound and its reflection off the
ground, each through its own delay as it changes, so the pitch falls as it passes; over
the grass the ground takes away much of the middle and the treble, the more the farther
away, while over the yard its reflection adds to the direct sound. The stills give the
level heard at each moment, the loudest (L_Amax) and the sound exposure level (SEL) of
the pass, as simulated, how close the vehicle comes and how far the pitch of its direct
sound falls; the vehicle's trajectory, edited in the editor's Movement panel, is driven
as edited.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

DEMO = preset(
    name="passby",
    example="roadside",
    title=N_("A car or a truck passing the meadow"),
    description=__doc__,
    script=Path(__file__),
    # The script's car, and its vehicles each in a part of their own.
    defaults={"sources": "car", "sources_play": "in-turn"},
    renamed={"scene": (("--from", str),), "sources": (("--vehicle", str),)},
    steps={"feeds": ("sound",), "simulate": ("scene",), "render": ("walk",)},
)

if __name__ == "__main__":
    sys.exit(DEMO.main())
