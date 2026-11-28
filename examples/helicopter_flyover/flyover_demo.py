"""A helicopter flying over the meadow 150 m up, heard right under its track and 150 m to
its left and to its right, its pitch falling as it passes (Doppler); and a video that
follows.

This is the demo of the helicopter_flyover scene (``acoustic-spaces demo scene.yaml``,
This scene in the editor): a preset of it, which keeps the command lines of the script it
was. The helicopter is the scene's source ``helicopter``, of a rotorcraft model: an Airbus
Helicopters AS350 B3 at its maximum mass, flying level along its trajectory at 63 m/s over
the ground (``--speed``, in km/h), facing the way it flies. Its rotors are trimmed for
level flight at its airspeed (its speed over the ground and the light wind against it),
and their sound is each harmonic of each rotor's blade passing frequency with a pattern of
its own, from the blades' steady lift, drag and thickness, and broadband noise by Pegg's
method (``--broadband bpm``: each blade section's own noise in the flow past it, Brooks,
Pope and Marcolini's). Each early path plays what the rotors send the way it leaves the
helicopter, at the time it leaves; the model gives the helicopter the sound power and the
directivity of that sound, for the levels the simulation reports and for its late part,
which the meadow alone does not give. ``--input helicopter=WAV`` plays a dry recording of
yours instead, sent with that directivity. Run it from the folder of the
helicopter_flyover example (``acoustic-spaces examples --copy . helicopter_flyover && cd
helicopter_flyover``), in the Python environment acoustic-spaces is installed in, with
ffmpeg on the PATH::

    python flyover_demo.py all      # every step, leaving out what is there already
    python flyover_demo.py feeds    # its sound sent straight down, dry, to hear
    python flyover_demo.py render   # it every 4 m along its flight, at every listener
    python flyover_demo.py video    # demo/flyover_binaural.mp4 (an animation), its stills

The script's steps run as the ones that took them over: ``helicopter`` the feeds,
``scene`` the simulation, ``walk`` the mixes; ``--from`` names the scene. Each run goes to
a folder in ./demo named after what it changes (demo/flyover_broadband-bpm_binaural/),
with the video, its audio and its thumbnail beside it; ``--speed 150,250`` plays a pass at
each speed in one video. Use headphones; ``--hrtf me.sofa`` plays it through your own or
measured HRTFs instead of a spherical head.

Each listener hears the helicopter through its direct sound and its reflection off the
grass, each through its own delay as it changes, so the pitch of its blades' tones falls
as it passes, and each carries what the rotors send its way at the time it leaves. The
wind against the helicopter bends the sound it sends ahead up and the sound it leaves
behind down, and the eddies make it flicker. The stills give the level heard at each
moment, the loudest (L_Amax) and the sound exposure level (SEL) of the pass, as
simulated. The rotors' model is louder than this helicopter's noise certificate: in the
certification's own flyover it gives an SEL of 88.0 dB(A), where the AS350 B3 is
certified at 84.6, nearly all of it the main rotor's broadband noise by Pegg's method;
with --broadband bpm it gives 80.1, the tail rotor's tones then nearly all of it (see
the validation). It flies one steady level flight: the slap of the blades meeting their
own wake, as a helicopter descending makes, is left out.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

DEMO = preset(
    name="flyover",
    example="helicopter_flyover",
    title=N_("A helicopter flying over"),
    description=__doc__,
    script=Path(__file__),
    renamed={"scene": (("--from", str),)},
    steps={"feeds": ("helicopter",), "simulate": ("scene",), "render": ("walk",)},
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
