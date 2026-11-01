"""Road works in the street: a 1650 kg breaker on an excavator breaking up the road, a
diesel generator and a dumper reversing to them with its alarm on, heard binaurally on
the pavement, on a balcony and in the garden behind the noise barrier, and a video that
tours them.

The machines are the example's own (the sources `breaker`, `generator`, `dumper` and
`dumper_alarm`, left out of its simulations until ticked), each with the sound power
per octave band of a row of Defra's noise database for construction sites (2005, the
data of BS 5228-1 Annex C): its level 10 m away plus 28 dB. Their sound is made to that
power: the breaker's blows, 480 a minute (--blows), and the generator's and the
dumper's engines firing; the alarm beeps at 1200 Hz, 1.5 times a second, 107 dB(A) 1.2
m in front of it (SAE J994 Type B), at the dumper's back facing the works.
`--input NAME=WAV` plays a dry recording of yours instead. Run it from the folder of the
street example (`acoustic-spaces examples --copy . street && cd street`), in the Python
environment acoustic-spaces is installed in, with ffmpeg on the PATH:

    python works_demo.py scene      # scene_works.yaml: the street with its works alone
    python works_demo.py feeds      # demo_works/feeds: what each machine plays, dry
    python works_demo.py simulate   # the scene, into demo_works/results
    python works_demo.py render     # every listener, binaural
    python works_demo.py video      # demo_works/tour.wav, the stills, demo_works/tour.mp4

or `python works_demo.py all`, every step, leaving out what is there already; the
editor's Simulation > Run demo… runs it too. Use headphones; `--hrtf me.sofa` plays it
through your own or measured HRTFs instead of a spherical head.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos import works

DEMO = works.demo(Path(__file__), "street")

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
