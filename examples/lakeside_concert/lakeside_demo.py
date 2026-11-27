"""A song played on the lakeside concert example's stage, heard in the audience, behind
the stage, from a boat and in the village across the lake, and a video that tours them.

This is the demo of the lakeside's scene (``acoustic-spaces demo scene.yaml``, This
scene in the editor): a preset of it, which keeps the command lines of the script it
was. The stage's sound system, two line arrays and subwoofers, plays the scene's
recording, by default "Spicy Funk Cake" by the Rhythmussportgruppe, a nine-piece funk
band, recorded for the German Acoustical Society (DEGA) at Hochschule Düsseldorf and
published under CC BY 4.0 (https://doi.org/10.5281/zenodo.3601032): ``--recording
great_british_summer`` (``--song``, the script's own flag) another free song, or
``--recording song.wav --bpm 120`` (``--music``) a song of your own and its tempo. The
tour is the scene's: each place for phrases of 8 bars, and at a bedroom window across
the lake the song in the evening's weather, which bends the sound down to the water,
then in still air, where it goes straight (a second simulation of the scene without
its atmosphere), then in the evening's again. Run it from the folder of the lakeside
concert example (``acoustic-spaces examples --copy . lakeside_concert && cd
lakeside_concert``), in the Python environment acoustic-spaces is installed in, with
ffmpeg on the PATH::

    python lakeside_demo.py all        # every step, leaving out what is there already
    python lakeside_demo.py recording  # the song (56 MB)
    python lakeside_demo.py render     # every place, in the evening and in still air
    python lakeside_demo.py video      # the video, its audio and its stills

Each stop is turned up to the loudness of the loudest (K-weighted), and its still says
by how much; the window in still air keeps the gain of the window in the evening's
weather, so the difference the weather makes is heard as the simulation gives it. Each
place's mix is shifted earlier by the time the sound takes to reach it (1.6 s across
the lake), so the beat carries on across the cuts. ``--listeners
bedroom_window,audience`` tours those places alone, in that order, each with its stops
of the tour, and back to the first at the end; ``--tour`` gives a tour of your own
(LISTENER[:still][:LENGTH]; the script's listener:weather:phrases still reads), and
``--last-stop`` (``--last-phrases``) how long the last stop lasts: two phrases by
default, ``end`` (0) to the end of the song. ``--sources subs``
plays the subwoofers alone (``main_left``, ``main_right`` and ``subs``, or ``PA`` for
all three, the default). The script's steps run as the ones that took them over:
``music`` the recording and the feeds, ``scenes`` the simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

#: The weathers of the script's ``--tour``: the evening's is the scene's own.
WEATHERS = ("evening", "still", "")


def _tour(text: str) -> str:
    """A ``--tour`` of the script's, ``listener:weather:phrases,...`` with the weather
    evening or still and one phrase by default, as the demo of any scene takes it
    (``listener[:still][:length]``); one in that form as it is."""
    parts = []
    for part in text.split(","):
        fields = [field.strip() for field in part.split(":")]
        if len(fields) > 1 and fields[1] not in WEATHERS:  # listener:length
            parts.append(":".join(fields))
            continue
        fields += [""] * (3 - len(fields))
        if fields[1] == "evening":
            fields[1] = ""
        if not fields[2]:
            fields[2] = "1"
        elif fields[2] == "0":  # the script's to the end of the song
            fields[2] = "end"
        parts.append(":".join(fields))
    return ",".join(parts)


def _last(phrases: str) -> str:
    """The script's ``--last-phrases``, as ``--last-stop`` takes it: 0 to the end."""
    return "end" if float(phrases) == 0.0 else phrases


DEMO = preset(
    name="lakeside",
    example="lakeside_concert",
    title=N_("A concert across a lake"),
    description=__doc__,
    script=Path(__file__),
    # The script's last stop, of its tour or of one given, lasted two phrases.
    defaults={"last_stop": "2"},
    renamed={
        "scene": (("--from", str),),
        "recording": (("--song", str), ("--music", str)),
        "tour": (("--tour", _tour),),
        "last_stop": (("--last-phrases", _last),),
    },
    steps={"recording": ("music",), "feeds": ("music",), "simulate": ("scenes",)},
)

if __name__ == "__main__":
    sys.exit(DEMO.main())
