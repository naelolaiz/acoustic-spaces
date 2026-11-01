"""Doc 29's twin jet JETW taking off from the aerodrome or landing on it, heard
binaurally under its departure 4 km from its start of roll, beside the runway 450 m from
its centre line and under its approach 2 km before the threshold, its pitch falling as
it passes (Doppler); and a video that follows.

The jet is the example's source `departure` or `arrival` (--vehicle picks it; the
arrival is left out of the example's own simulations until ticked), each a point whose
walk gives its ground track. The
demo flies it along that track as ECAC Doc 29 flies JETW (its Appendix B, at Volume 3's
default weights, in the scene's air): the departure by the ICAO-A noise abatement
procedure from its start of roll, the arrival down a 3° approach to touch down where its
walk first comes down to the runway, then its reverse thrust and roll-out. The heights,
speeds and thrust along the way are Doc 29's, and so is how long the flight takes (the
example's departure 95 s, its arrival 93 s), so there is no --speed. Its sound power
comes from its NPD curves (its maximum level 1000 ft under it at the flight's highest
thrust, with its spectral class's spectrum) and its directivity from its
installation correction (engines under the wings). Its sound is noise with that
spectrum, louder and softer as the thrust changes; `--input departure=WAV` plays a dry
recording of yours instead. Run it from the folder of the airport example
(`acoustic-spaces examples --copy . airport && cd airport`), in the Python environment
acoustic-spaces is installed in, with ffmpeg on the PATH:

    python flight_demo.py aircraft  # demo/departure/engines.wav: its engines along the flight, dry
    python flight_demo.py scene     # scene_demo.yaml: the jet alone, its flight as keyframes
    python flight_demo.py walk      # the jet every 4 m along its flight, at every listener
    python flight_demo.py video     # demo/tour.wav, demo/tour.mp4 (an animation), stills

or `python flight_demo.py all --vehicle arrival`, every step, leaving out what is there
already; the editor's Simulation > Run demo… runs it too. Use headphones; `--hrtf
me.sofa` plays it through your own or measured HRTFs instead of a spherical head.

Each listener hears the jet's direct sound and its reflection off the ground, each
through its own delay as it changes, so each shifts in pitch as it passes; the late
sound is crossfaded from place to place. The jet's sound fades in and out over a second
where its flight starts and ends. The captions give the level heard at each moment, the
loudest (L_Amax) and the sound exposure level (SEL) of the pass, as simulated;
validation/aircraft_noise.py compares them with Doc 29's own sum over the flight.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos import passby
from acoustic_spaces.demos.aircraft import ARRIVAL, DEPARTURE

DEMO = passby.demo(
    Path(__file__), "airport", "A jet taking off and landing", __doc__, (DEPARTURE, ARRIVAL)
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
