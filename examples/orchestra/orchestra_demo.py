"""Beethoven's 8th symphony in the concert hall example, heard from its four seats, and a
video that tours them through three movements.

This is the demo of the orchestra's scene (``acoustic-spaces demo scene.yaml``, This
scene in the editor): a preset of it, which keeps the command lines of the script it
was. The recording is the TU Berlin's anechoic one (Böhm, Ackermann and Weinzierl,
Orchester Wiener Akademie; CC BY 4.0, https://doi.org/10.14279/depositonce-6729.2),
``beethoven_8`` in the recordings catalogue: 61 tracks, one per player, which the
example puts on the stage as 61 sources, each playing its track as recorded, so that the
balance is the orchestra's. The three movements it has (I, II and IV) come as
movement1.zip, movement2.zip and movement4.zip: the ``recording`` step reads the 61
tracks of each movement it plays out of the zip on the TU Berlin's server, by HTTP range
requests, without keeping the zip (2.0 GB for the three), or out of a copy downloaded
before (``--zips FOLDER``, the example's folder by default), and keeps them in the
recordings folder for every scene. Run this from the folder of the example
(``acoustic-spaces examples --copy . orchestra && cd orchestra``), in the Python
environment acoustic-spaces is installed in, with ffmpeg on the PATH::

    python orchestra_demo.py all        # every step, leaving out what is there already
    python orchestra_demo.py recording  # the 61 tracks of each movement (2.0 GB)
    python orchestra_demo.py simulate   # the hall with the orchestra and its four seats
    python orchestra_demo.py render     # each seat, in each movement, binaural
    python orchestra_demo.py video      # the video, its audio and its stills

The tour is the scene's: each movement from two to four of the seats in turn, each seat
for an even share of it, the cut moved to the quietest moment within 10 s, so that it
falls between phrases, with a 1 s crossfade; each movement is a part of the video,
after a title card, the hall's reverberation dying away at its end. Every seat's mix is
shifted earlier by the time the orchestra's sound takes to reach it, so the music runs
on across a cut, and all stops keep one gain: the hall's differences are heard as the
simulation gives them. ``--piece IV,I`` plays the finale, then the first movement (the
script's ``--movements 4,1`` still reads); ``--listeners stalls-rear,conductor`` plays
each movement at the seats of its tour among those, in that order, or at all of them
when its tour has none of them; ``--tour I/conductor,IV/stalls-rear`` gives a tour of
your own (the script's ``movement:seat,...`` still reads). ``--sources Winds`` plays the
winds alone (by section, ``Strings/Violins I`` to ``Percussion/Timpani``, or by player,
``fl1``); a movement every picked player rests through (the trumpets and the timpani in
II) is left out, unless ``--input`` gives one of them a file of yours, which it plays in
every movement. The simulation keeps every player and seat, so another pick of them
needs no new one. The script's ``tracks`` step runs as ``recording``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

#: The movements by their numbers, as the script's ``--movements`` and ``--tour`` gave
#: them, and their names in the recording.
NUMERALS = {"1": "I", "2": "II", "4": "IV"}


def _movement(text: str) -> str:
    """A movement as the script's ``--movements`` gave it (4), as ``--piece`` takes it
    (IV); one in that form as it is."""
    return NUMERALS.get(text.strip(), text.strip())


def _tour(text: str) -> str:
    """A ``--tour`` of the script's, ``movement:seat,...``, as the demo of any scene
    takes it, ``PIECE/seat,...`` (a movement the recording lacks kept as its number, which
    the run refuses); one in that form as it is."""
    parts = []
    for part in text.split(","):
        movement, colon, seat = (field.strip() for field in part.partition(":"))
        if colon and movement.isdigit() and seat[:1].isalpha() and ":" not in seat:
            part = f"{_movement(movement)}/{seat}"
        parts.append(part)
    return ",".join(parts)


DEMO = preset(
    name="orchestra",
    example="orchestra",
    title=N_("Beethoven's 8th in the concert hall"),
    description=__doc__,
    script=Path(__file__),
    # The script read the zips downloaded before from the example's folder, and kept one
    # gain for every seat.
    defaults={"zips": Path("."), "levels": "as-simulated"},
    renamed={"piece": (("--movements", _movement),), "tour": (("--tour", _tour),)},
    steps={"recording": ("tracks",)},
)

if __name__ == "__main__":
    sys.exit(DEMO.main())
