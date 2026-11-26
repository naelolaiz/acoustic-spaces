"""A walk through the gallery and a look round, heard binaurally: a guide talks while a
visitor walks away from them, down the long arm of the L-shaped gallery and round the
corner, and another stands in front of them and turns their head; and a video that
follows.

This is the demo of the gallery's scene (``acoustic-spaces demo scene.yaml``, This
scene in the editor) with the walker and the one turning alone, ``--listeners
visitor,standing``: a preset of it, which keeps the command lines of the script it
was. Run it from the folder of the gallery example (``acoustic-spaces examples --copy .
gallery && cd gallery``), in the Python environment acoustic-spaces is installed in,
with ffmpeg on the PATH::

    python walk_demo.py all        # every step, leaving out what is there already
    python walk_demo.py recording  # the talk (reads the corpus's archive up to it: 16 MB)
    python walk_demo.py render     # the walk and the turn, every mover on one clock
    python walk_demo.py video      # the video, its audio and its stills

The guide's voice is real: a talk recorded in an anechoic room for the ACE challenge
corpus (Eaton, Gaubitch, Moore and Naylor, Imperial College London; CC BY 4.0), a man
describing his way to work through Delft; ``--recording ace_f1`` (or ``--talker F1``,
the script's own flag) picks another of its 14 talkers, the guide's speech spectrum a
woman's or a man's with it, and ``--vocal-effort raised`` (``--voice``) how loud the
guide talks (ISO 9921). The walk is the scene's visitor's; the turn is the scene's
listener standing's, 4 m in front of the guide, a quarter at a time and held, through
the head tracker: ``--stand 4,15`` stands it elsewhere, facing the guide, and ``--turn
0:0,6:180`` turns it otherwise (yaw in degrees at each time in seconds). The stills say
the speech level there and the STI of the guide's speech with its rating. The script's
steps run as the ones that took them over: ``speech`` the recording, ``scene`` the
simulation, ``walk`` and ``turn`` render.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

#: The scene's listener that turns its head, which the script's ``--stand`` and
#: ``--turn`` gave no name.
STANDING = "standing"


def _standing(value: str) -> str:
    """A ``--stand`` or ``--turn`` of the script's, for the listener standing: as the
    demo of any scene takes it, ``standing=...``; one that names its listener as it is."""
    return value if "=" in value else f"{STANDING}={value}"


DEMO = preset(
    name="walk",
    example="gallery",
    title=N_("A walk through the gallery"),
    description=__doc__,
    script=Path(__file__),
    # The script's walk and look round, at one gain.
    defaults={"listeners": f"visitor,{STANDING}", "levels": "as-simulated"},
    renamed={
        "scene": (("--from", str),),
        "recording": (("--talker", lambda code: f"ace_{code.lower()}"),),
        "stand": (("--stand", _standing),),
        "turn": (("--turn", _standing),),
    },
    steps={"recording": ("speech",), "simulate": ("scene",), "render": ("walk", "turn")},
)

if __name__ == "__main__":
    sys.exit(DEMO.main())
