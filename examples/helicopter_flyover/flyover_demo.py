"""A helicopter flying over the meadow 150 m up, heard right under its track and 150 m to
its left and to its right, its pitch falling as it passes (Doppler); and a video that
follows.

The helicopter is the example's own (the source `helicopter`): an Airbus Helicopters
AS350 B3 at its maximum mass, flying level along its walk at 63 m/s over the ground
(--speed, in km/h), facing the way it flies. Its sound is its two rotors' from
acoustic_spaces.rotorcraft, trimmed for level flight at its airspeed (its speed over the
ground and the light wind against it): each harmonic of each rotor's blade passing
frequency with a pattern of its own, from the blades' steady lift, drag and thickness,
and broadband noise by Pegg's method. Each early path plays what the rotors send the way
it leaves the helicopter, and the demo's scene gives the helicopter the sound power and
the directivity of that sound, for its late part and the levels it reports; `--input
helicopter=WAV` plays a dry recording of yours instead, sent with that directivity. Run
it from the folder of the helicopter_flyover example (`acoustic-spaces examples --copy .
helicopter_flyover && cd helicopter_flyover`), in the Python environment acoustic-spaces
is installed in, with ffmpeg on the PATH:

    python flyover_demo.py helicopter  # demo/helicopter/below.wav: its sound straight down
    python flyover_demo.py scene       # scene_demo.yaml: the helicopter alone, at --speed
    python flyover_demo.py walk        # it every 4 m along its flight, at every listener
    python flyover_demo.py video       # demo/tour.wav, demo/tour.mp4 (an animation), stills

or `python flyover_demo.py all`, every step, leaving out what is there already; the
editor's Simulation > Run demo… runs it too. Use headphones; `--hrtf me.sofa` plays it
through your own or measured HRTFs instead of a spherical head.

Each listener hears the helicopter through its direct sound and its reflection off the
grass, each through its own delay as it changes, so the pitch of its blades' tones falls
as it passes, and each carries what the rotors send its way at the time it leaves. The
wind against the helicopter bends the sound it sends ahead up and the sound it leaves
behind down, and the eddies make it flicker. The captions give the level heard at each
moment, the loudest (L_Amax) and the sound exposure level (SEL) of the pass, as
simulated. The rotors' model is louder than this helicopter's noise certificate: in the
certification's own flyover it gives an SEL of 88.0 dB(A), where the AS350 B3 is
certified at 84.6, nearly all of it the main rotor's broadband noise (see the
validation). It flies one steady level flight: the slap of the blades meeting their own
wake, as a helicopter descending makes, is left out.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos import passby
from acoustic_spaces.demos.helicopter import HELICOPTER_VEHICLE

DEMO = passby.demo(
    Path(__file__), "helicopter_flyover", "A helicopter flying over", __doc__, HELICOPTER_VEHICLE
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
