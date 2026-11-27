"""Road works in the street: a 1650 kg breaker on an excavator breaking up the road, a
diesel generator and a dumper reversing to them with its alarm on, heard binaurally on
the pavement, on a balcony and in the garden behind the noise barrier, and a video that
tours them.

This is the demo of the street's scene (``acoustic-spaces demo scene.yaml --sources
"Road works"``, This scene in the editor): a preset of it, which keeps the command lines
of the script it was. The machines are the scene's own sources ``breaker``,
``generator``, ``dumper`` and ``dumper_alarm`` (the group ``Road works``), left out of
its simulations until picked; they play together, ``--sources breaker,dumper`` those
alone, and ``--sources-play in-turn`` each in a part of its own, one after the other in
one video. Each has a model in the scene. The machines have the sound power per octave
band of a row of Defra's noise database for construction sites (2005, the data of BS
5228-1 Annex C), its level 10 m away plus 28 dB, and their sound is made to that power:
the breaker's blows, 480 a minute (``--blows``), and the generator's and the dumper's
engines firing. The alarm beeps at 1200 Hz, 1.5 times a second, 107 dB(A) 1.2 m in
front of it (SAE J994 Type B), at the dumper's back facing the works. They go on all
through the run, ``--seconds`` at each listener. ``--input NAME=WAV`` plays a dry
recording of yours instead. ``--indoors`` hears them in the room behind each listener
that stands in front of a building's wall (the pavement, 2 m in front of the houses,
and the balcony, 1 m in front of the offices): through its closed window, from the
level 2 m in front of the wall, and the rumble the breaker's and the dumper's vibration
makes in it, or either alone (``airborne_and_rumble``, ``airborne``, ``rumble``; the
script's ``works_and_rumble`` and ``works`` read as the first two); what is felt at each
listener, the ground's or the house's floor's vibration, is a track for a bass shaker
too (``.shaker.wav`` beside each mix, and the run's ``<run>_shaker.wav``). Run it from
the folder of the street example (``acoustic-spaces examples --copy . street && cd
street``), in the Python environment acoustic-spaces is installed in, with ffmpeg on
the PATH::

    python works_demo.py all       # every step, leaving out what is there already
    python works_demo.py feeds     # what each machine and the alarm play, dry
    python works_demo.py simulate  # the street with the works taken in
    python works_demo.py render    # every listener, with one gain for all
    python works_demo.py video     # demo/works_binaural.mp4, its audio, the stills

The script's ``scene`` step runs as ``simulate``, which writes the scenes it simulates
to the work folder, ``demo``; ``--from`` names the scene, and ``--tour`` the listeners
in turn. Each run's mixes and stills go to a folder there named after its settings
(``demo/works_binaural/``), with the video beside it. Use headphones;
``--hrtf me.sofa`` plays it through your own or measured HRTFs instead of a spherical
head.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

#: The road works, the scene's group ``Road works``: what the script played, every one
#: in one mix by default, each in a part of its own with ``--sources-play in-turn``.
WORKS = ("breaker", "generator", "dumper", "dumper_alarm")
#: What the script's ``--indoors`` took for the works, as the demo's takes it: the
#: works through the window is the airborne sound.
INDOORS = {"works_and_rumble": "airborne_and_rumble", "works": "airborne"}


def old_indoors(value: str) -> str:
    """A value of the script's ``--indoors`` as the demo's takes it (:data:`INDOORS`)."""
    return INDOORS.get(value, value)


DEMO = preset(
    name="works",
    example="street",
    title=N_("Road works in the street"),
    description=__doc__,
    script=Path(__file__),
    # The script's works, all together, or with --sources-play in-turn each alone.
    defaults={"sources": ",".join(WORKS)},
    renamed={"scene": (("--from", str),), "indoors": (("--indoors", old_indoors),)},
    steps={"simulate": ("scene",)},
)

__all__ = ["DEMO", "INDOORS", "WORKS", "old_indoors"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
