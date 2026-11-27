"""A song played by a band on the stadium example's stage and through its sound system,
heard at each of its listeners, and a video that tours them.

This is the demo of the stadium's scene (``acoustic-spaces demo scene.yaml``, This scene
in the editor): a preset of it, which keeps the command lines of the script it was. The
scene's recording is the song, one of the recordings catalogue's songs with their stems
(``--recording``, or the script's ``--song``): "Spicy Funk Cake" by the
Rhythmussportgruppe (composed by Niklas Dahlheimer), recorded for the sound database of
the DEGA (German Acoustical Society) at Hochschule Düsseldorf, CC BY 4.0,
https://doi.org/10.5281/zenodo.3601032, or "Francium" or "Rubidium" by
Software-Entwicklungskit from the album "Everything is Free", CC0 1.0,
https://github.com/ichbinsoftware/everythingisfree; or a song of your own (``--recording
song.wav``, or the script's ``--music``, with ``--bpm`` its tempo, ``--downbeat`` its
first downbeat and ``--stem`` its stems). The sound system (main arrays, subwoofers,
fills, towers and delays) plays the song's stereo master; the monitors play mixes of its
stems: the singer's wedges the vocal, the drum fill the drums and bass, the side fills a
mix for the whole stage; and the band on the stage plays its own parts: the drum kit the
drums, the singer the vocal, the guitarist's amplifier the guitar (the synth lead, for
the album's songs). Run it from the folder of the stadium example (``acoustic-spaces
examples --copy . stadium_concert && cd stadium_concert``), in the Python environment
acoustic-spaces is installed in, with ffmpeg on the PATH::

    python stadium_demo.py all        # every step, leaving out what is there already
    python stadium_demo.py recording  # the master and the stems (42 to 310 MB)
    python stadium_demo.py simulate   # the stadium with its band, binaural
    python stadium_demo.py feeds      # one signal per loudspeaker and player
    python stadium_demo.py render     # every listener, binaural, with one gain for all
    python stadium_demo.py video      # demo/stadium_binaural.mp4 (its audio, its stills)

The script's steps run as the ones that took them over: ``music`` the recording,
``band`` the simulation (the band is the scene's own now); ``--from`` names the scene.
Each run goes to a folder in ./demo named after what it changes
(demo/stadium_francium_binaural/), with the video, its audio and its thumbnail beside
it, so that a run of another song or with other settings writes over none of them. Use
headphones; ``--hrtf me.sofa`` plays through your own or measured HRTFs instead of a
spherical head (the simulation takes it too). ``--input drum_kit=drums.wav`` has a
source play a file of yours instead of its feed, at its sound power. The stills draw the
stadium from above with its crowd, a dot for every few people, without the view from
the side that the stills of an outdoor scene have (``--hide none`` draws it, ``--hide
side,audience`` leaves the crowd out too).

Each source's feed is brought to one gated RMS (the master's left and right sharing one
gain, a monitor's mix of stems as a whole) and plays at its sound power in the scene
(the main arrays' 134 dB, the fills' 116 to 130 dB, the monitors' and the band's), so
the balance between them is the scene's. The video plays the song once, moving between
listeners on the scene's tour for the song, its stops in phrases of 8 bars from the
first downbeat and the last to the end, with a 1 s crossfade; each listener's mix is
shifted earlier by the time the sound takes to reach them, so that the beat carries on
across the cut. On stage it is 6 to 18 dB(A) louder than in the audience, so each stop
is turned up to the loudness of the loudest one (K-weighted) and its still says by how
much; the mixes of a run keep one gain for all. ``--listeners east_stand,mix_position``
mixes those listeners alone, and the video tours them in that order, each for its stop
of the tour and the last to the end of the song; ``--tour`` gives a tour of your own
(the script's ``listener:phrases,...`` still reads). ``--sources PA`` mixes the sound
system alone (or ``Monitors``, ``Band``, a group such as ``PA/Delays`` or a source by
its name, comma separated), from the same simulation and feeds; ``--sources-play
in-turn`` plays the sound system, the monitors and the band one after another, each
alone, in one video. The scene's helicopter, left out of the concert's simulations,
flies over it in helicopter_demo.py, the stadium's other demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_


def _tour(text: str) -> str:
    """A ``--tour`` of the script's, ``listener:phrases,...`` with the last stop to the
    end of the song whatever it says, as the demo of any scene takes it; one with a
    stop in that demo's own form (``:still``, ``12s``, ``end``) as it is."""
    parts = [[field.strip() for field in part.split(":")] for part in text.split(",")]
    try:
        if any(len(fields) != 2 or float(fields[1]) < 0.0 for fields in parts):
            return text
    except ValueError:
        return text
    stops = [f"{name}:{phrases}" for name, phrases in parts[:-1] if float(phrases) > 0.0]
    return ",".join([*stops, f"{parts[-1][0]}:end"])


DEMO = preset(
    name="stadium",
    example="stadium_concert",
    title=N_("A band in the stadium"),
    description=__doc__,
    script=Path(__file__),
    # The script's tour ended at the song's end, whatever its last stop said.
    defaults={"last_stop": "end", "hide": "side"},
    renamed={
        "scene": (("--from", str),),
        "recording": (("--song", str), ("--music", str)),
        "tour": (("--tour", _tour),),
    },
    steps={"recording": ("music",), "simulate": ("band",)},
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
