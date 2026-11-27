"""A helicopter flying low over the stadium concert halfway through the song, heard at
each listener over the band and the sound system, its pitch falling as it passes
(Doppler); and a video that tours them, as the stadium demo's does.

This is the demo of the stadium's scene (``acoustic-spaces demo scene.yaml``, This scene
in the editor) with the helicopter taken in, on one clock with the concert: a preset of
it, which keeps the command lines of the script it was. The helicopter is the scene's
source ``helicopter``, of a rotorcraft model, left out of the concert's own simulations:
an Airbus Helicopters AS350 B3 at its maximum mass, flying level at 63 m/s, straight
over the stadium's long axis from behind the stage to past the east stand, 1200 m. It
flies 150 m up, as a police or a television helicopter given leave may: without leave,
the EU's rules of the air keep a flight by sight (VFR) over an open-air assembly of
persons 300 m above the highest obstacle within 600 m (SERA.5005(f)), here the
stadium's roof, 30 m up, so ``--height 330``, which takes every place of its trajectory
to that height. The editor shows its trajectory, and what Edit movement… changes there
the demo flies. It sets off as the scene has it, so that it is over the centre spot
97.5 s into the run, halfway through "Spicy Funk Cake"; ``--at 180`` has it over the
centre spot 180 s into the song as the sound system plays it, and ``--start`` sets when
it sets off. Its sound is its two rotors', sent each way down each early path, as in
the helicopter_flyover example; its model gives it the sound power and the directivity
of that sound, for the levels the simulation reports and for its late part.

Run it from the folder of the stadium example (``acoustic-spaces examples --copy .
stadium_concert && cd stadium_concert``), in the Python environment acoustic-spaces is
installed in, with ffmpeg on the PATH::

    python helicopter_demo.py all        # every step, leaving out what is there already
    python helicopter_demo.py recording  # the song's master and its stems
    python helicopter_demo.py simulate   # the stadium with its band and the helicopter
    python helicopter_demo.py feeds      # one signal per loudspeaker and player, the rotors'
    python helicopter_demo.py render     # every listener, the helicopter every 4 m
    python helicopter_demo.py video      # demo/crossing_binaural.mp4, its audio, the stills

The script's steps run as the ones that took them over: ``flight`` and ``scene`` the
simulation, ``helicopter`` the feeds, ``walk`` and ``mix`` the mixes; ``--from`` names
the scene. It needs no run of the stadium demo first: the scene, the song and the
helicopter are simulated and mixed together, the ground and the sky widened to take the
flight. The script's ``--stadium``, ``--band``, ``--results`` and ``--flight`` (the
scenes and the simulation it read and wrote) are no longer taken: the scenes a run
simulates go to the work folder. Each run goes to a folder in ./demo named after what it
changes (demo/crossing_height-330_binaural/), with the video, its audio and its
thumbnail beside it. Use headphones. The stills draw the stadium as the stadium demo's
do, the helicopter at their edge, with how far it is, while it is past the stadium.
``--listeners east_stand,mix_position`` mixes those listeners alone, and the video tours
them in that order; ``--tour`` gives a tour of your own (the script's
``listener:phrases,...`` still reads). ``--sources PA`` has it fly over the sound system
alone (or ``Monitors``, ``Band``, a group such as ``PA/Delays`` or a source by its name,
comma separated); the helicopter always plays, and ``--sources helicopter`` has it fly
alone, to hear it without the concert. ``--speed``, ``--broadband``, ``--step``
and ``--input helicopter=WAV`` are the flyover demo's.

Both play at their simulated levels, as every source of the scene does: each
loudspeaker's feed stands for its sound power, and the helicopter's sound is what its
rotors send. The video turns each stop up to the loudness of the loudest, as the
stadium demo's does. The stadium's air is still: no wind or eddies bend the
helicopter's sound. Its rotors' downwash is left out: flying past at this speed, their
wake trails behind them and sinks a few metres a second (1.6 m/s through the main
rotor's disc in its trim), far above the crowd. With Pegg's broadband noise, the
default, the rotors' model is louder than this helicopter's noise certificate, and
with --broadband bpm quieter (see the helicopter_flyover example).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

from acoustic_spaces.demos import IN_TURN, given_names, sibling
from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

stadium = sibling(__file__, "stadium_demo")

HERE = Path(__file__).resolve().parent
#: The scene's helicopter, which always plays.
HELICOPTER = "helicopter"
#: What the crossing plays by default: the concert's sources and the helicopter.
CONCERT = ("PA", "Monitors", "Band")


def over_the_centre_s(args: argparse.Namespace) -> float:
    """How long the scene's helicopter takes from where it sets off to over the centre
    spot (nearest to it in plan, between the places of its trajectory), at ``--speed``
    (km/h) when given, else its trajectory's (s)."""
    import numpy as np

    from acoustic_spaces.scene import Source
    from acoustic_spaces.walkthrough import at_speed, knots

    path = Path(args.scene)
    if not path.is_absolute() and not path.exists():
        path = HERE / path
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    found = next((s for s in data.get("sources") or [] if s.get("name") == HELICOPTER), None)
    if found is None or not found.get("trajectory"):
        raise ValueError(f"{path.name} has no source {HELICOPTER!r} with a trajectory")
    # Its places and timing alone: not its model, whose sound power takes some seconds.
    plain = {key: value for key, value in found.items() if key in ("name", "position")}
    plain["trajectory"] = {**found["trajectory"], "start_s": 0.0}
    source = Source.model_validate(plain)
    if getattr(args, "speed", None) is not None:
        source = at_speed(source, float(args.speed) / 3.6)
    path_of = knots(source)
    plan = path_of.positions[:, :2]
    legs = np.diff(plan, axis=0)
    along = np.clip(
        -np.sum(plan[:-1] * legs, axis=1) / np.maximum(np.sum(legs**2, axis=1), 1e-12), 0, 1
    )
    k = int(np.argmin(np.linalg.norm(plan[:-1] + along[:, None] * legs, axis=1)))
    times = path_of.times
    return float(times[k] + along[k] * (times[k + 1] - times[k]))


OLD_FLAGS = (
    (
        "--at",
        {
            "dest": "old_at",
            "type": float,
            "help": "when the helicopter is over the centre spot, in seconds into the song as "
            "the sound system plays it, as the script took it (--start sets when it sets "
            "off; default: as the scene has it, 97.5 s, halfway through Spicy Funk Cake)",
        },
    ),
)


def reads_old(args: argparse.Namespace) -> str | None:
    """The helicopter always plays: ``--sources`` with it beside the concert's sources
    it picks (in each part of its own, with ``--sources-play in-turn``); the script's
    ``--at`` as ``--start``, so that the helicopter is over the centre spot then. What
    is wrong with them, or None."""
    names = given_names(getattr(args, "sources", None)) or list(CONCERT)

    def flies(name: str) -> bool:
        return HELICOPTER in name.lower().split("+")

    if getattr(args, "sources_play", None) == IN_TURN:
        # Each part of the concert with the helicopter over it, none of it alone.
        parts = [name for name in names if name.lower() != HELICOPTER] or [HELICOPTER]
        names = [name if flies(name) else f"{name}+{HELICOPTER}" for name in parts]
    elif not any(flies(name) for name in names):
        names = [*names, HELICOPTER]
    args.sources = ",".join(names)
    at = getattr(args, "old_at", None)
    if hasattr(args, "old_at"):
        delattr(args, "old_at")
    if at is None:
        return None
    if getattr(args, "start", None) is not None:
        return "--at and --start both say when the helicopter flies: give one or the other"
    if not math.isfinite(at) or at < 0.0:
        return f"--at is a time into the song (s), not {at!r}"
    try:
        takes = over_the_centre_s(args)
    except (OSError, ValueError) as exc:
        return f"--at: {exc}"
    if at < takes:
        return (
            f"--at is {at:g} s, but the helicopter takes {takes:.1f} s to get over the "
            f"stadium: give --at {math.ceil(takes)} or later"
        )
    args.start = round(at - takes, 3)
    return None


DEMO = preset(
    name="crossing",
    example="stadium_concert",
    title=N_("A helicopter over the stadium concert"),
    description=__doc__,
    script=Path(__file__),
    # The concert and the helicopter on one clock, the tour going on as the song plays.
    defaults={
        "sources": ",".join((*CONCERT, HELICOPTER)),
        "timeline": "shared",
        "last_stop": "end",
        "hide": "side",
    },
    renamed={
        "scene": (("--from", str),),
        "recording": (("--song", str), ("--music", str)),
        "tour": (("--tour", stadium._tour),),
    },
    steps={
        "simulate": ("flight", "scene"),
        "feeds": ("helicopter",),
        "render": ("walk", "mix"),
    },
    old_flags=OLD_FLAGS,
    reads_old=reads_old,
    builds_on="stadium",
)

__all__ = ["DEMO", "over_the_centre_s", "reads_old"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
