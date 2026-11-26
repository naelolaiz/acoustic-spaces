"""A walk through the gallery and a look round, heard binaurally: a guide talks while a
visitor walks away from them, down the long arm of the L-shaped gallery and round the
corner, then stands in front of them and turns their head; and a video that follows.

The guide's voice is real: a talk recorded in an anechoic room for the ACE challenge
corpus (Eaton, Gaubitch, Moore and Naylor, Imperial College London; CC BY 4.0), a man
describing his way to work through Delft (--talker picks another of its 14 talkers).
Run it from the folder of the gallery example (`acoustic-spaces examples --copy .
gallery && cd gallery`), in the Python environment acoustic-spaces is installed in, with
ffmpeg on the PATH:

    python walk_demo.py speech  # the talk (reads the corpus's archive up to it: 16 MB)
    python walk_demo.py scene   # scene_demo.yaml: the guide with a normal voice
    python walk_demo.py walk    # responses every 25 cm along the walk, the talk through them
    python walk_demo.py turn    # the visitor stands 4 m from the guide and turns round
    python walk_demo.py video   # demo/walk_M4_binaural.mp4 (an animation), its audio, stills

or `python walk_demo.py all`, every step, leaving out what is there already; the
editor's Simulation > Run demo… runs it too. The speech goes to ./demo (--work), and
each run's scene, walk, head turn and stills to a folder there named after its settings
(the talker, what changed, the HRTFs: demo/walk_M4_binaural/), with the video, its
audio and its thumbnail beside it, so that a run with other settings writes over none
of them. Use headphones; `--hrtf me.sofa` plays it through your own or measured HRTFs instead of a
spherical head. `--input guide=talk.wav` has the guide say a recording of yours instead
(mixed down to mono); the voice's level and
STI in the captions are still the scene's (--voice), as the one gain of the video plays
any recording as loud. A talk shorter than the walk and the turn need (about a minute,
longer than some of the corpus's takes) is said again from its start, AGAIN_S after it
ends, as often as they need.

`--listeners` picks the parts and their order, by the listener each follows: `visitor`
the walk, `standing` the head turn. `--listeners standing,visitor` turns the head
first and walks after, `--listeners visitor` walks alone; the talk goes on from one
part into the next, and the video plays those alone.

The walk is the example's visitor's: it sets off 3 m in front of the guide and faces the
way it goes, so the guide is behind it down the long arm; at the end it turns into the
short arm, where the guide is out of sight and the voice reaches it bent round the
corner (the scene simulates first-order diffraction). The demo hears the visitor only;
the gallery's two other listeners stand where they are and are left out. Edit the
visitor's trajectory in the editor's Movement panel and the demo walks it as edited;
with it turned off (Moving) there is no walk to play, and the demo says so. Then the
turn: the visitor stands 4 m in front of the guide, where the scene's listener
standing is (left out of its own simulations; --stand moves it), faces them and turns
left a quarter at a time, holding each (--turn): the voice moves to the right ear,
behind, to the left ear and back in front. The turn goes through the engine's head
tracker, scripted: the place's response in Ambisonics of order 7, the sound field turned against
the head every 256 samples and decoded to the ears (acoustic_spaces.headtracking).
Through a spherical head or a measured one it comes within half a decibel, in each
ear and octave, of rendering the ears again at every angle.

A spherical head, the default, tells left from right but hardly front from behind, as
it has no outer ears: the guide behind sounds much as in front. Measured HRTFs
(--hrtf) tell them apart.

The scene gives the guide a voice as IEC 60268-16 describes it: the long-term spectrum
of male (or female) speech at 60 dB(A) 1 m in front for a normal voice (--voice), and
a voice's directivity, all round in the bass and towards the front in the treble. The
captions state the speech level this gives at the visitor's ears and the speech
transmission index (STI) in the silent gallery, both simulated; one gain plays the
whole video, so the voice is as much quieter round the corner as the simulation says.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from acoustic_spaces import fetch
from acoustic_spaces.demos import (
    Demo,
    Option,
    Source,
    Step,
    chosen_order,
    common_options,
    input_option,
    listeners_option,
    pipeline,
    product,
    run_folder,
    sibling,
    sources_options,
)
from acoustic_spaces.demos.audio import envelope, mono, quiet_moment, read, write
from acoustic_spaces.recordings import ACE, ACE_DOI

# --------------------------------------------------------------------------- sources

#: The corpus's talkers (5 women, 9 men): each one's take 4, about a minute on how they
#: get to work.
TALKERS = tuple(f"F{k}" for k in range(1, 6)) + tuple(f"M{k}" for k in range(1, 10))
#: MB of the archive read before each talker's take is out: it is read as it comes.
TALK_MB = {
    "M4": 16,
    "M2": 43,
    "M8": 49,
    "F2": 49,
    "M1": 79,
    "M6": 87,
    "M5": 99,
    "M3": 110,
    "F4": 120,
    "F3": 125,
    "M7": 133,
    "M9": 135,
    "F5": 136,
    "F1": 145,
}
CREDITS = f"""\
Speech: the ACE challenge corpus, anechoic speech, by James Eaton, Nikolay D. Gaubitch,
Alastair H. Moore and Patrick A. Naylor (Imperial College London, 2015),
https://doi.org/{ACE_DOI}, CC BY 4.0

Credit for sharing a render (the video, say):

Speech: ACE challenge corpus by J. Eaton, N. D. Gaubitch, A. H. Moore, P. A. Naylor
(CC BY 4.0, doi:{ACE_DOI}), placed in a simulated gallery. License:
https://creativecommons.org/licenses/by/4.0/
"""

# --------------------------------------------------------------------------- the visit

#: The speech level 1 m in front of a talker, dB(A), by vocal effort (IEC 60268-16).
VOICES = {"relaxed": 54.0, "normal": 60.0, "raised": 66.0, "loud": 72.0}
#: A voice's directivity per octave band, 63 Hz to 8 kHz: all round in the bass,
#: towards the front in the treble.
VOICE_PATTERN = [
    "omni",
    "omni",
    "omni",
    "subcardioid",
    "subcardioid",
    "cardioid",
    "cardioid",
    "cardioid",
]
#: The scene's listener where the visitor stands to look round, left out of its own
#: simulations, and the head's turn there: yaw (degrees, turning left; 0 faces the
#: guide) at each time (s).
STANDING = "standing"
TURN = [
    (0.0, 0.0),
    (4.0, 0.0),
    (10.0, 90.0),
    (14.0, 90.0),
    (20.0, 180.0),
    (24.0, 180.0),
    (30.0, 270.0),
    (34.0, 270.0),
    (40.0, 360.0),
]
#: How long the visitor listens once the walk is over, and the turn once it is round
#: (s), each up to a pause in the talk within PAUSE_S.
LINGER_S = 4.0
PAUSE_S = 1.5
#: The silence before a talk too short for the parts is said again (s).
AGAIN_S = 2.0
#: The rooms' tail kept after each part's talk ends, the fade at its end, and the
#: silence between the parts (s).
TAIL_S = 1.6
FADE_S = 0.6
GAP_S = 0.6
WALK = "walk.json"
TURNING = "turn.json"
#: The listener of the example that walks.
VISITOR = "visitor"
#: The parts a run can play, in the demo's order, each by its step's name (which names
#: its folder and its sound) and the listener it follows: the visitor's walk, then the
#: head turn where the listener standing is.
PARTS = {"walk": VISITOR, "turn": STANDING}
#: The file of each part's facts, in its folder.
FACTS = {"walk": WALK, "turn": TURNING}


def speech_path(args) -> Path:
    return args.work / "speech" / f"{args.talker}.wav"


def speech_download(args) -> str:
    """What ``speech`` downloads: the archive as far as the talk (:data:`TALK_MB`)."""
    return (
        f"the talk of {args.talker} out of the ACE corpus's speech archive "
        f"({TALK_MB[args.talker]} of its 148 MB, read as far as the talk)"
    )


def speech(args) -> None:
    """The talk of ``--talker`` out of the ACE corpus's speech archive (148 MB, read as
    it comes; the take is 2 to 11 MB), its transcript and the credits."""
    folder = args.work / "speech"
    members = {
        f"Speech/{args.talker}s4.wav": folder / f"{args.talker}.wav",
        f"Speech/{args.talker}s4.txt": folder / f"{args.talker}.txt",
    }
    print(f"reading {ACE} for the talk of {args.talker}")
    fetch.extract_tar(ACE, members)
    (folder / "SOURCES.txt").write_text(CREDITS, encoding="utf-8")
    print(f"wrote {speech_path(args)} and its transcript")


def guide_input(args) -> Path | None:
    """The file ``--input guide=WAV`` gives the guide to say, or None."""
    return (getattr(args, "inputs", None) or {}).get("guide")


def plays(args) -> dict[str, str]:
    """What the guide says unless ``--input`` gives it a file."""
    return {"guide": f"the talk of ACE talker {args.talker}"}


def played_by(args) -> str:
    """What the guide says, as walk.json and turn.json keep it: the file ``--input``
    gives it, or the talk of ``--talker``."""
    file = guide_input(args)
    return str(file) if file is not None else f"ACE {args.talker}"


def talk(args) -> tuple[np.ndarray, int]:
    """The talk the guide says, as recorded: ``--input guide=``'s or ``--talker``'s
    take, mixed down to mono."""
    path = guide_input(args) or speech_path(args)
    if not path.exists():
        raise SystemExit(f"error: no {path}; run first: {DEMO.command('speech', args)}")
    x, rate = read(path)
    return mono(x), rate


def talk_name(args) -> str:
    what = guide_input(args)
    return what.name if what is not None else f"the talk of {args.talker}"


def listeners(args) -> list[str]:
    """The listeners a run can play (``--listeners``), a part of the video each: the
    visitor walking, and standing, where the visitor turns its head."""
    return list(PARTS.values())


def order_of(args) -> list[str]:
    """The parts a run plays, by their steps' names, in its order: as ``--listeners``
    orders their listeners; the walk, then the turn, when it is not given."""
    heard = chosen_order(args, listeners(args))
    return [part for name in heard for part, who in PARTS.items() if who == name]


def parts_words(args) -> str:
    """The parts a run plays, in its order, in words: ``the walk and the turn``."""
    return " and ".join(f"the {part}" for part in order_of(args))


def length_of(args, room, part: str) -> float:
    """How long the visitor walks (``walk``) or turns its head (``turn``), in s, before
    it listens LINGER_S longer."""
    if part == "walk":
        return float(walk_points_of(room).times[-1])
    return turn_of(args)[-1][0]


def needed_s(args, room) -> float:
    """How long a talk the parts the run plays need (s): each, the visitor's LINGER_S
    after it and the PAUSE_S a cut may wait for a pause in the talk."""
    order = order_of(args)
    lengths = sum(length_of(args, room, part) for part in order)
    return lengths + len(order) * (LINGER_S + PAUSE_S) + 1.0


def said(args, room) -> tuple[np.ndarray, int, int]:
    """The talk as the guide says it (:func:`talk`), and how many times: once, or again
    from its start AGAIN_S after it ends, as often as the parts need
    (:func:`needed_s`)."""
    x, rate = talk(args)
    if not len(x):
        raise SystemExit(f"error: {talk_name(args)} is empty")
    length_s, need_s = len(x) / rate, needed_s(args, room)
    times = max(1, math.ceil((need_s + AGAIN_S) / (length_s + AGAIN_S)))
    if times == 1:
        return x, rate, 1
    gap = np.zeros(int(round(AGAIN_S * rate)), dtype=x.dtype)
    return np.concatenate([x, *([gap, x] * (times - 1))]), rate, times


def say_again(args, room, spoken: np.ndarray, rate: int, times: int) -> None:
    """Say so when the guide says the talk more than once (:func:`said`)."""
    if times > 1:
        once_s = (len(spoken) / rate - (times - 1) * AGAIN_S) / times
        take = "take" if len(order_of(args)) > 1 else "takes"
        print(
            f"{talk_name(args)} is {once_s:.0f} s long, and {parts_words(args)} {take} "
            f"{needed_s(args, room):.0f} s: the guide says it {times} times, "
            f"{AGAIN_S:g} s apart"
        )


def female(args) -> bool:
    return args.talker.startswith("F")


# --------------------------------------------------------------------------- the scene


def voice_level(spectrum: str, pattern: list[str], at_1m_a: float) -> float:
    """The sound power (dB re 1 pW) of a talker with ``spectrum`` and ``pattern`` whose
    speech is ``at_1m_a`` dB(A) 1 m in front of them in free field."""
    from acoustic_spaces.directivity import PATTERNS
    from acoustic_spaces.parameters import a_weighted
    from acoustic_spaces.scene.model import Source

    a = np.array([PATTERNS[name] for name in pattern])
    forward = 1.0 / (a**2 + (1.0 - a) ** 2 / 3.0)  # on-axis gain of a + (1 - a) cos
    source = Source(name="talker", position=(0.0, 0.0, 0.0), level_db=0.0, spectrum=spectrum)
    at_1m = source.power_db("octave") + 10.0 * np.log10(forward / (4.0 * math.pi))
    return round(at_1m_a - float(a_weighted(at_1m)), 1)


def scene(args) -> None:
    """``--scene`` (scene_demo.yaml) from ``--from`` (scene.yaml): the gallery with the
    guide's voice as IEC 60268-16 gives it (``--voice``, the talker's spectrum) and its
    directivity, the visitor's walk, and the scene's listener where it stands to look
    round taken in (moved to ``--stand`` and turned to the guide, when given)."""
    base = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    base["name"] = "Gallery walkthrough with a guide talking"
    spectrum = "speech_female" if female(args) else "speech_male"
    (guide,) = [s for s in base["sources"] if s["name"] == "guide"]
    guide.update(
        {
            "level_db": voice_level(spectrum, VOICE_PATTERN, VOICES[args.voice]),
            "spectrum": spectrum,
            "directivity": {"pattern": list(VOICE_PATTERN)},
        }
    )
    standing = next((lis for lis in base["listeners"] if lis["name"] == STANDING), None)
    if standing is None:
        raise SystemExit(f"error: {args.source} has no listener {STANDING!r}")
    standing.pop("enabled", None)
    if args.stand:
        x, y = stand_of(args)
        gx, gy = guide["position"][:2]
        standing["position"] = [x, y, standing["position"][2]]
        standing["orientation"] = {"yaw": round(math.degrees(math.atan2(gy - y, gx - x)), 2)}
    pipeline.write_scene(
        args.scene,
        base,
        "# The gallery example with the guide's voice as a talker's, and where the visitor\n"
        "# stands to look round, written by walk_demo.py.\n",
        args.source.parent,
        made_from=args.source,
    )
    print(f"wrote {args.scene}: the guide at {guide['level_db']} dB of sound power")


def stand_of(args) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in str(args.stand).split(","))
    except ValueError:
        raise SystemExit(f"error: --stand takes x,y in metres, not {args.stand!r}") from None
    return x, y


def turn_of(args) -> list[tuple[float, float]]:
    """The turn's (time s, yaw degrees) from ``--turn`` (``t:yaw,...``), yaw 0 facing
    the guide."""
    if not args.turn:
        return list(TURN)
    try:
        return [tuple(float(v) for v in part.split(":")) for part in args.turn.split(",")]
    except ValueError:
        raise SystemExit(f"error: --turn takes t:yaw,..., not {args.turn!r}") from None


def room_of(args):
    from acoustic_spaces import Room

    if not args.scene.exists():
        raise SystemExit(f"error: no {args.scene}; run first: {DEMO.command('scene', args)}")
    return Room.from_yaml(args.scene)


#: The Ambisonics order the turn is rendered in before the ears.
TURN_ORDER = 7


def hrtf_of(args, rate: int):
    """The HRTFs the parts go through: ``--hrtf``, or the spherical head model."""
    from acoustic_spaces.spatial.hrtf import read_sofa, spherical_head

    if not args.hrtf:
        return spherical_head(rate)
    hrtf = read_sofa(args.hrtf)
    return hrtf if hrtf.sample_rate == rate else hrtf.resampled(rate)


def label_of(args) -> str:
    """The mixes' name: binaural, or binaural-<name> through ``--hrtf``."""
    return pipeline.hrtf_label("binaural", args.hrtf)


# --------------------------------------------------------------------------- the talk


def visitor_of(scene):
    """The visitor of ``scene`` (a :class:`~acoustic_spaces.scene.Scene`), with its walk."""
    visitor = next((lis for lis in scene.listeners if lis.name == VISITOR), None)
    if visitor is None:
        raise SystemExit(f"error: the scene has no listener {VISITOR!r} to walk")
    return visitor


def walk_points_of(room):
    from acoustic_spaces.walkthrough import trajectory_points

    return trajectory_points(visitor_of(room.scene))


def pace_of(visitor) -> str:
    """How fast the visitor goes, for the captions: the walk's speed along its points, or
    the average along keyframes (their length over their time)."""
    from acoustic_spaces.walkthrough import knots

    if visitor.trajectory.keyframes is None:
        return f"at {visitor.trajectory.speed_m_s:g} m/s"
    path = knots(visitor)
    length = float(np.linalg.norm(np.diff(path.positions, axis=0), axis=1).sum())
    return f"at {length / path.duration_s:.1f} m/s on average"


def segments(args, room) -> dict[str, tuple[float, float]]:
    """The part of the talk each part of the run plays, by its step's name, in the
    run's order (s, from and to): the first from the talk's start, each next from where
    the one before ends, each to a pause near its end, after the visitor has listened
    LINGER_S longer; in the talk as said (:func:`said`), as often as they need it."""
    x, rate, _ = said(args, room)
    step = 0.1
    env = envelope(x, rate, step)
    out, start = {}, 0.0
    for part in order_of(args):
        stop = quiet_moment(env, step, start + length_of(args, room, part) + LINGER_S, PAUSE_S)
        out[part], start = (start, stop), stop
    return out


def parts_before(args, part: str) -> list[str]:
    """The parts the run plays before ``part`` (one it plays), by their steps' names."""
    order = order_of(args)
    return order[: order.index(part)]


def own_before(part: str) -> list[str]:
    """The parts before ``part`` in the demo's own order: the walk, then the turn."""
    return list(PARTS)[: list(PARTS).index(part)]


def facts_after(args, part: str) -> dict:
    """What ``part``'s facts keep of the parts before it (``after``), when they are not
    those of the demo's own order: files from before ``--listeners`` came in that
    order, and the run of every part writes them as it did then."""
    after = parts_before(args, part)
    return {} if after == own_before(part) else {"after": after}


def guide_of(room):
    return next(s for s in room.scene.sources if s.name == "guide")


def facts_of(room, responses) -> dict:
    """At each of ``responses`` (places of the walk, or where the visitor stands): the
    guide's speech level (dB(A)) and the STI, as simulated (no background noise)."""
    from acoustic_spaces.parameters import a_weighted

    guide = guide_of(room)
    power = guide.power_db("octave")
    levels, sti = [], []
    for response in responses:
        levels.append(round(float(a_weighted(response.levels(power))), 2))
        sti.append(float(response.speech(power, None, talker=guide.talker).sti))
    return {"level_a_db": levels, "sti": sti}


def walk(args) -> None:
    """The walk, when the run plays the visitor (``--listeners``): responses every 25 cm
    along it, the late part traced every 2 m, the walk's part of the talk
    (:func:`segments`) played through them; ``walk/walk.<label>.wav`` (binaural, or
    binaural-<name> through ``--hrtf``) and walk.json (the places, the level and STI at
    each, the part of the talk), in the run's folder."""
    from acoustic_spaces.spatial import Binaural
    from acoustic_spaces.walkthrough import render_trajectory, simulate_trajectory

    if "walk" not in order_of(args):
        print(f"walk: --listeners leaves out the {VISITOR}: no walk to play")
        return
    room = room_of(args)
    x, rate, times = said(args, room)
    say_again(args, room, x, rate, times)
    start, stop = segments(args, room)["walk"]
    print("simulating the walk (the late part takes the longest)")
    walked = simulate_trajectory(room, VISITOR, progress=print)
    rate_out = walked.result.settings.sample_rate
    ears = Binaural(hrtf_of(args, rate_out))
    wet = render_trajectory(
        walked, {"guide": (x[int(start * rate) : int(stop * rate)], rate)}, ears
    )
    out = run_folder(args) / "walk"
    write(out / f"walk.{label_of(args)}.wav", wet, rate_out)
    pipeline.write_json(
        out / WALK,
        {
            "talker": args.talker,
            "plays": played_by(args),
            "talk_s": [start, stop],
            **facts_after(args, "walk"),
            "times": walked.points.times.tolist(),
            "positions": walked.points.positions.tolist(),
            "yaws": yaws_of(walked, room).tolist(),
            **facts_of(room, [walked.result["guide", name] for name in walked.names]),
        },
    )
    print(
        f"wrote {out}: {walked.duration_s:.1f} s of walking, the talk from {start:.1f} s "
        f"to {stop:.1f} s"
    )


def yaws_of(walked, room) -> np.ndarray:
    """The way the visitor faces at each place of the walk: its own yaw all along when it
    does not face forward."""
    if walked.points.yaws is not None:
        return walked.points.yaws
    return np.full(len(walked.points.times), visitor_of(room.scene).orientation.yaw)


def turning(times, yaws, facing: float, length_s: float, every_s: float = 0.5):
    """The head tracker's ``head`` (yaw at ``t`` s from ``times`` and ``yaws``, pitch and
    roll 0) that says, at most every ``every_s`` of the run, which way the head faces
    then (:func:`~acoustic_spaces.walkthrough.facing_name`, ``facing`` the yaw it starts
    from) and how far the turn has come: ``guide -> standing facing 90.0 deg: turning the
    head, 12.0 of 40.0 s``."""
    import time

    from acoustic_spaces.walkthrough import facing_name

    said = [-math.inf]

    def head(t: float) -> tuple[float, float, float]:
        yaw = float(np.interp(t, times, yaws))
        now = time.monotonic()
        if now - said[0] >= every_s:
            said[0] = now
            print(
                f"guide -> {facing_name(STANDING, facing + yaw)}: turning the head, "
                f"{t:.1f} of {length_s:.1f} s",
                flush=True,
            )
        return yaw, 0.0, 0.0

    return head


def turn(args) -> None:
    """The turn, when the run plays standing (``--listeners``): the visitor where the
    scene's listener standing is, simulated there once, the turn's part of the talk
    (:func:`segments`) through its response in Ambisonics, turned against the head as
    ``--turn`` scripts it and decoded to the ears (the engine's head tracker);
    ``turn/turn.<label>.wav`` and turn.json."""
    from acoustic_spaces import Room, simulate
    from acoustic_spaces.headtracking import HeadTrackedBinaural
    from acoustic_spaces.render import resample
    from acoustic_spaces.spatial import Ambisonics

    if "turn" not in order_of(args):
        print(f"turn: --listeners leaves out {STANDING}: no turn to play")
        return
    room = room_of(args)
    x, rate, times = said(args, room)
    say_again(args, room, x, rate, times)
    start, stop = segments(args, room)["turn"]
    standing = [lis for lis in room.scene.listeners if lis.name == STANDING]
    if not standing:
        raise SystemExit(f"error: {args.scene} has no listener {STANDING!r}; run: scene")
    alone = Room.from_scene(room.scene.model_copy(update={"listeners": standing}), reuse=room)
    print("simulating the place the visitor looks round from")
    result = simulate(alone, progress=print)
    response = result["guide", STANDING]
    rate_out = result.settings.sample_rate
    seed = result.settings.seed if result.settings.seed is not None else response.seed
    field = response.spatial_impulse_response(Ambisonics(TURN_ORDER), rate_out, seed)
    tracker = HeadTrackedBinaural(field, rate_out, hrtf=hrtf_of(args, rate_out))
    keys = turn_of(args)
    times, yaws = [t for t, _ in keys], [yaw for _, yaw in keys]
    dry = resample(x[int(start * rate) : int(stop * rate)], rate, rate_out)
    facing = standing[0].orientation.yaw
    print(f"turning the head through {len(keys)} keys")
    length_s = (len(dry) + tracker.taps - 1) / rate_out  # the talk and the room's tail
    wet = tracker.run(dry, head=turning(times, yaws, facing, length_s))
    out = run_folder(args) / "turn"
    write(out / f"turn.{label_of(args)}.wav", wet, rate_out)
    pipeline.write_json(
        out / TURNING,
        {
            "talker": args.talker,
            "plays": played_by(args),
            "talk_s": [start, stop],
            **facts_after(args, "turn"),
            "facing": facing,
            "keys": [list(key) for key in keys],
            "times": times,
            "positions": [list(standing[0].position)] * len(keys),
            "yaws": [facing + yaw for yaw in yaws],
            **facts_of(room, [response] * len(keys)),
        },
    )
    print(f"wrote {out}: the talk from {start:.1f} s to {stop:.1f} s")


# --------------------------------------------------------------------------- the video


def part_audio(path: Path, talk_s: float) -> tuple[np.ndarray, int]:
    """A part's sound to its talk's end and TAIL_S of the room after it, faded out."""
    x, rate = read(path)
    x = x[: int((talk_s + TAIL_S) * rate)]
    fade = int(FADE_S * rate)
    x[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None] ** 2
    return x, rate


def video(args) -> None:
    """The run's audio (``<run>.wav``: its parts in its order, the walk and the turn by
    default, a breath of silence between them: one gain, the loudest peak at -1 dBFS),
    and its video (``<run>.mp4``): the gallery from above, the visitor walking or
    turning, and what they hear, with a scrolling spectrogram (ffmpeg); stills of its
    moments."""
    from acoustic_spaces.demos.pictures import save, settings_lines

    pictures = sibling(__file__, "walk_pictures")
    label = label_of(args)
    order = order_of(args)
    parts, pieces, rate = [], [], None
    for name in order:
        folder = part_folder(args, name) or run_folder(args) / name
        path = folder / f"{name}.{label}.wav"
        if not path.exists():
            raise SystemExit(f"error: no {path}; run first: {DEMO.command(name, args)}")
        facts = pipeline.read_json(folder / FACTS[name])
        start, stop = facts["talk_s"]
        x, rate = part_audio(path, stop - start)
        x = np.atleast_2d(x.T).T
        if pieces:
            pieces.append(np.zeros((int(GAP_S * rate), pieces[0].shape[1])))
        at = sum(len(piece) for piece in pieces) / rate
        pieces.append(x)
        parts.append((name, facts, at, at + len(x) / rate))
    audio = np.concatenate(pieces)
    audio *= 10.0 ** (-1.0 / 20.0) / float(np.abs(audio).max())
    sound = product(args, ".wav")
    write(sound, audio, rate)
    print(f"wrote {sound} ({len(audio) / rate:.1f} s)")
    pace = pace_of(visitor_of(room_of(args).scene)) if "walk" in order else ""
    scene_data = yaml.safe_load(args.scene.read_text(encoding="utf-8"))
    drawing = pictures.Pictures(
        scene_data,
        parts,
        voice=(args.voice, VOICES[args.voice]),
        credit=credit_of(args),
        hrtf=args.hrtf,
        pace=pace,
        tool=settings_lines(DEMO, args, scene_data),
    )
    save(drawing.thumbnail(), product(args, "_thumbnail.png"))
    moments = drawing.moments()
    pipeline.animation(
        run_folder(args),
        len(audio) / rate,
        drawing.frame,
        sound,
        out=product(args, ".mp4"),
        stills=[(name, t + 0.5) for name, t, _ in moments],
        no_mp4=args.no_mp4,
        chapters=[f"{int(t // 60)}:{int(t % 60):02d} {title}" for _, t, title in moments],
        shown=lambda t: drawing.shown(t, VISITOR, STANDING),
    )


def credit_of(args) -> str:
    """The stills' credit line: the ACE corpus's, or the file the guide says instead."""
    file = guide_input(args)
    if file is not None:
        return f"Playing guide: {file.name}"
    return (
        f"Speech: ACE challenge corpus, talker {args.talker} (Eaton, Gaubitch, Moore, "
        f"Naylor, Imperial College London), CC BY 4.0 · doi.org/{ACE_DOI}"
    )


def speech_done(args) -> bool:
    """Whether the talk is there, or not needed (``--input guide=...``)."""
    return guide_input(args) is not None or speech_path(args).exists()


def part_done(args, name: str) -> bool:
    """Whether the walk's or the turn's sound the run plays is there
    (:func:`part_folder`), or the run does not play that part (``--listeners``)."""
    return name not in order_of(args) or part_folder(args, name) is not None


def part_folder(args, name: str) -> Path | None:
    """The folder of the walk's or the turn's sound the run plays, or None when it is
    not there: the run's own, or that of a run whose sound of it is the same, the part
    of the talk too: the run of the demo's own order of the parts and the run of this
    part alone (``--listeners``), and the run with the same settings up to that step of
    each (:meth:`~acoustic_spaces.demos.Demo.upstream`: the walk of a run with another
    ``--turn`` is the same). It is taken when it is made for the ears this run asks for
    (``--hrtf``), from what the guide says now (files written before they kept "plays"
    said the talk of their "talker"), from that run's scene as it is now, after the
    same parts (:func:`facts_after`) and, after the turn, the same ``--turn``."""
    after, seen = parts_before(args, name), set()
    alone = DEMO.sibling(args, listeners=PARTS[name])
    for each in (args, DEMO.sibling(args, listeners=None), alone):
        for run in (each, DEMO.upstream(each, name)):
            folder = run_folder(run) / name
            if folder in seen:
                continue
            seen.add(folder)
            path = folder / FACTS[name]
            if not path.exists() or not (folder / f"{name}.{label_of(run)}.wav").exists():
                continue  # not made yet, or only through another head
            if run.scene.exists() and path.stat().st_mtime < run.scene.stat().st_mtime:
                continue  # the scene changed since: the walk edited, say
            facts = pipeline.read_json(path)
            if facts.get("plays", f"ACE {facts.get('talker')}") != played_by(run):
                continue  # the guide said something else
            if facts.get("after", own_before(name)) != after or (
                "turn" in after and turn_of(run) != turn_of(args)
            ):
                continue  # another part of the talk
            return folder
    return None


def scene_done(args) -> bool:
    """Whether scene_demo.yaml is there, made from the example's scene ``--from`` names
    and newer than it (the visitor's walk edited in the editor makes it again)."""
    return pipeline.scene_made(args.scene, args.source)


def moving(args) -> list[str]:
    """Who the demo plays moving: the visitor, when the run plays its walk
    (``--listeners``) and the scene gives it a walk that is on."""
    from acoustic_spaces.scene import Scene

    if "walk" not in order_of(args):
        return []
    try:
        visitor = visitor_of(Scene.from_yaml(args.source))
    except (Exception, SystemExit):
        return []
    return [VISITOR] if visitor.trajectory is not None and visitor.trajectory.enabled else []


def sources(args) -> list[Source]:
    """The sources a run can play (``--sources``): the guide, the gallery's only one."""
    return [Source("guide")]


def problems(demo, args) -> str | None:
    """The visitor must walk, when the run plays its walk (``--listeners``): the
    example's scene gives it a walk that is on."""
    from acoustic_spaces.scene import Scene

    source = Path(demo.resolved(args).source)
    if not source.exists():
        return f"no {source}"
    try:
        scene = Scene.from_yaml(source)
    except Exception as exc:  # the scene's own message
        return f"{source.name}: {exc}"
    if "walk" not in order_of(args):
        return None
    visitor = next((lis for lis in scene.listeners if lis.name == VISITOR), None)
    if visitor is None:
        return f"{source.name} has no listener {VISITOR!r} to walk"
    if visitor.trajectory is None:
        return (
            f"the {VISITOR} has no trajectory in {source.name}: give it one (trajectory:, or the "
            "editor's Movement panel)"
        )
    if not visitor.trajectory.enabled:
        return (
            f"the {VISITOR}'s trajectory is off in {source.name}: turn it on (Moving in the "
            "editor, or trajectory: enabled: true)"
        )
    return None


DEMO = Demo(
    name="walk",
    example="gallery",
    title="A walk through the gallery",
    description=__doc__,
    script=Path(__file__),
    steps=[
        Step(
            "speech",
            speech,
            "download the talk (16 to 145 MB of the corpus's 148 MB archive)",
            speech_done,
            speech_download,
        ),
        Step(
            "scene",
            scene,
            "scene_demo.yaml: the guide's voice, where the visitor looks round",
            scene_done,
        ),
        Step(
            "walk",
            walk,
            "simulate the walk and play the talk along it",
            lambda args: part_done(args, "walk"),
        ),
        Step(
            "turn",
            turn,
            "simulate where the visitor stands and turn its head",
            lambda args: part_done(args, "turn"),
        ),
        Step("video", video, "the run's video, its audio and the stills, named after the run"),
    ],
    options=[
        Option(
            "talker",
            "the ACE corpus's talker",
            "M4",
            "choice",
            TALKERS,
            ("speech", "walk", "turn", "video"),
            named="plays",
        ),
        Option(
            "voice",
            "how loud the guide speaks (IEC 60268-16)",
            "normal",
            "choice",
            tuple(VOICES),
            ("scene",),
        ),
        Option(
            "source",
            "the example's scene",
            Path("scene.yaml"),
            "file",
            steps=("scene",),
            flag="--from",
            filter="Scene files (*.yaml *.yml)",
        ),
        Option(
            "scene",
            "the scene with the guide's voice (default: scene_demo.yaml in the run's folder)",
            None,
            "path",
            in_run="scene_demo.yaml",
        ),
        Option(
            "stand",
            "x,y where the visitor stands to look round (m); by default the scene's listener"
            " 'standing'",
            None,
            steps=("scene",),
        ),
        Option(
            "turn",
            "t:yaw,... the head's yaw (degrees, turning left, 0 facing the guide) at each time (s)",
            None,
            steps=("turn",),
        ),
        listeners_option(
            ("walk", "turn", "video"),
            help="the parts to play, by the listener each follows, comma separated, in "
            f"their order: {VISITOR} (the walk), {STANDING} (the head turn); the talk "
            "goes on from one part into the next, each simulating its own listener "
            f"(default: {VISITOR},{STANDING}, the walk then the turn)",
        ),
        *[o for o in common_options() if o.name not in ("again", "format")],
        dataclasses.replace(
            input_option(("walk", "turn", "video")),
            help="a WAV file the guide says instead of the ACE talk, as guide=WAV: mixed "
            "down to mono, said again from its start when shorter than the parts need; "
            "the captions' level and STI are the scene's voice (--voice), as the video's "
            "one gain plays any recording as loud",
        ),
        *sources_options(
            help="the source that plays: guide, the gallery's only one (default: guide)",
            steps=("walk", "turn", "video"),
        ),
    ],
    scene_from="scene",
    problems=problems,
    plays=plays,
    moving=moving,
    listeners=listeners,
    sources=sources,
)


__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
