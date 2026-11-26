"""A choir and an organ in the church example, heard at its listeners, and a video that
tours them.

This is the demo of the church's scene (``acoustic-spaces demo scene.yaml`` with
``--sources Choir,"Organ pipes"``, This scene in the editor): a preset of it, which
keeps the command lines of the script it was. Nineteen singers stand in two rows in
front of the altar and sing the round "Dona nobis pacem"; the organ in the gallery above
the entrance plays it first, alone, and then accompanies them. The scene's recording,
``dona_nobis_pacem`` in the recordings catalogue, gives each of them its stem: each
singer's dry take (a recording booth, the microphone close) from VocalSet (CC BY 4.0),
brought to one tempo and, partly, to one tuning, and each side of the organ's divisions
playing a four-part harmonisation of the tune from the pipe organ samples of the
Versilian Community Sample Library (CC0). Run it from the folder of the church example
(``acoustic-spaces examples --copy . church && cd church``), in the Python environment
acoustic-spaces is installed in, with ffmpeg on the PATH::

    python church_demo.py all        # every step, leaving out what is there already
    python church_demo.py recording  # the takes and the organ samples (127 MB), built
    python church_demo.py simulate   # the church with its choir, organ and places
    python church_demo.py render     # every listener, binaural
    python church_demo.py video      # the video, its audio and its stills

The recording is built the first time into the recordings folder and kept for every
scene; its credits are the video's. The choir, the organ's pipes and four of the six
places the tour visits are left out of the church's own simulations: this demo takes
them in (``--sources Choir,"Organ pipes"`` by default), and simulates the church so,
with responses 9 s long (``--ir-length``), as the bass rings that long here. The scene
gives the sound powers, 88 dB re 1 pW for each singer (a choir singing mezzo forte), 105
dB for each side of the organ's manual and 102 dB for each pedal tower, where each
plays loudest; the quiet verse of the organ is 11 dB below its full one in its stems.

The music, at a crotchet a second: verse 1, the organ alone; verse 2, the choir alone;
verse 3, the choir with the organ on its quiet stops, slowing down into a held last
chord and the church's reverberation. The tour is the scene's: six places, each cut at a
breath of the music, with 1 s crossfades and one gain for all, so that the levels
between them are the church's; each mix is shifted earlier by the time the sound takes
to reach that place, so that the music runs on across a cut. ``--listeners
conductor,nave-back`` tours those places alone, in that order, each for its stop of the
tour and the last to the end; ``--tour`` gives a tour of your own (the script's
``listener:start_s,...`` still reads, from the music's start). ``--sources Choir`` plays
the choir alone, ``--sources "Organ pipes"`` the organ alone, ``--sources
singer_f1,organ_pedal_c`` those alone; the simulation keeps every source it took in, so
another pick of them needs no new one. ``--input singer_f1=take.wav`` has a singer or a
side of the organ play a file of yours instead of its stem. The script's steps run as
the ones that took them over: ``music`` the recording and the feeds, ``scene`` the
recording.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_


def _tour(text: str) -> str:
    """A ``--tour`` of the script's, ``listener:start_s,...`` (each place from its start
    until the next one's), as the demo of any scene takes it: each place for the
    seconds until the next one's start, from the music's start, the last to the end;
    one in that form as it is."""
    parts = [part.split(":") for part in text.split(",")]
    try:
        starts = [float(fields[1]) for fields in parts if len(fields) == 2]
    except ValueError:  # the demo's own form, LISTENER[:still][:LENGTH]
        return text
    if len(starts) != len(parts):
        return text
    ends = [*starts[1:], None]
    stops = []
    for k, (fields, end) in enumerate(zip(parts, ends, strict=True)):
        name = fields[0].strip()
        if end is None:
            stops.append(f"{name}:end")
            continue
        length = end - (0.0 if k == 0 else starts[k])
        stops.append(f"{name}:{length:g}s")
    return ",".join(stops)


DEMO = preset(
    name="church",
    example="church",
    title=N_("A choir and an organ in the church"),
    description=__doc__,
    script=Path(__file__),
    # The script's choir and organ, and the length of its responses.
    defaults={"sources": "Choir,Organ pipes", "ir_length": 9.0},
    renamed={"scene": (("--from", str),), "tour": (("--tour", _tour),)},
    steps={"recording": ("music", "scene"), "feeds": ("music",)},
)

if __name__ == "__main__":
    sys.exit(DEMO.main())
