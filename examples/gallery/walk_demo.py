"""A walk through the gallery and a look round, heard binaurally: a guide talks while a
visitor walks away from them, down the long arm of the L-shaped gallery and round the
corner, then stands in front of them and turns their head; and a video that follows.

The guide's voice is real: a talk recorded in an anechoic room for the ACE challenge
corpus (Eaton, Gaubitch, Moore and Naylor, Imperial College London; CC BY 4.0), a man
describing his way to work through Delft (--talker picks another of its 14 talkers).
Run it from the folder of the gallery example (`acoustic-spaces examples --copy .
gallery && cd gallery`), in the Python environment acoustic-spaces is installed in, with
ffmpeg on the PATH:

    python walk_demo.py speech  # the talk (reads the corpus's 148 MB speech archive)
    python walk_demo.py scene   # scene_demo.yaml: the guide with a normal voice
    python walk_demo.py walk    # responses every 25 cm along the walk, the talk through them
    python walk_demo.py turn    # the visitor stands 4 m from the guide and turns round
    python walk_demo.py video   # demo/tour.wav, demo/tour.mp4 (an animation), stills

or `python walk_demo.py all`, every step, leaving out what is there already; the
editor's Simulation > Run demo… runs it too. The scene goes beside the script
(scene_demo.yaml); the speech, the walk, the head turn and the video go to ./demo
(--work). Use headphones; `--hrtf me.sofa` plays it through your own or measured HRTFs instead of a
spherical head. `--input guide=talk.wav` has the guide say a recording of yours instead
(mono, as long as the walk and the turn need, about a minute); the voice's level and
STI in the captions are still the scene's (--voice), as the one gain of the video plays
any recording as loud.

The walk is the example's visitor's: it sets off 3 m in front of the guide and faces the
way it goes, so the guide is behind it down the long arm; at the end it turns into the
short arm, where the guide is out of sight and the voice reaches it bent round the
corner (the scene simulates first-order diffraction). The demo hears the visitor only;
the gallery's two other listeners stand where they are and are left out. Edit the walk
in the editor's Movement panel and the demo walks it as edited; with the visitor's walk
turned off (Moves) there is no walk to play, and the demo says so. Then the turn: the visitor stands
4 m in front of the guide (--stand), faces them and turns left a quarter at a time,
holding each (--turn): the voice moves to the right ear, behind, to the left ear and
back in front. The turn goes through the engine's head tracker,
scripted: the place's response in Ambisonics of order 7, the sound field turned against
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
    Step,
    common_options,
    input_option,
    pipeline,
    sibling,
)
from acoustic_spaces.demos.audio import envelope, mono, quiet_moment, read, write

# --------------------------------------------------------------------------- sources

ACE = "https://zenodo.org/api/records/6257551/files/ACE_Corpus_Speech.tbz2/content"
ACE_DOI = "10.5281/zenodo.6257551"
#: The corpus's talkers (5 women, 9 men): each one's take 4, about a minute on how they
#: get to work.
TALKERS = tuple(f"F{k}" for k in range(1, 6)) + tuple(f"M{k}" for k in range(1, 10))
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
#: Where the visitor stands to look round (x, y; m), and the head's turn: yaw (degrees,
#: turning left; 0 faces the guide) at each time (s).
STAND = (16.0, 4.0)
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
#: The rooms' tail kept after each part's talk ends, the fade at its end, and the
#: silence between the parts (s).
TAIL_S = 1.6
FADE_S = 0.6
GAP_S = 0.6
WALK = "walk.json"
TURNING = "turn.json"
#: The listener of the example that walks.
VISITOR = "visitor"


def speech_path(args) -> Path:
    return args.work / "speech" / f"{args.talker}.wav"


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
    path = guide_input(args) or speech_path(args)
    if not path.exists():
        raise SystemExit(f"error: no {path}; run first: {DEMO.command('speech', args)}")
    x, rate = read(path)
    return mono(x), rate


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
    directivity, the visitor's walk, and where it stands to look round (``--stand``)."""
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
    visitor = next(lis for lis in base["listeners"] if lis["name"] == VISITOR)
    x, y = stand_of(args)
    gx, gy = guide["position"][:2]
    yaw = round(math.degrees(math.atan2(gy - y, gx - x)), 2)
    standing = {
        "name": "standing",
        "position": [x, y, visitor["position"][2]],
        "orientation": {"yaw": yaw},
    }
    base["listeners"] = [lis for lis in base["listeners"] if lis["name"] != "standing"]
    base["listeners"].append(standing)
    pipeline.write_scene(
        args.scene,
        base,
        "# The gallery example with the guide's voice as a talker's, and where the visitor\n"
        "# stands to look round, written by walk_demo.py.\n",
        args.source.parent,
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
    from acoustic_spaces.walkthrough import walk_points

    return walk_points(visitor_of(room.scene))


def pace_of(visitor) -> str:
    """How fast the visitor goes, for the captions: the walk's speed along its points, or
    the average along keyframes (their length over their time)."""
    from acoustic_spaces.walkthrough import knots

    if visitor.walk.keyframes is None:
        return f"at {visitor.walk.speed_m_s:g} m/s"
    path = knots(visitor)
    length = float(np.linalg.norm(np.diff(path.positions, axis=0), axis=1).sum())
    return f"at {length / path.duration_s:.1f} m/s on average"


def cuts(args, room) -> tuple[float, float]:
    """Where the talk passes from the walk to the turn and where it ends (s): at a pause
    near the end of each, after the visitor has listened LINGER_S longer."""
    x, rate = talk(args)
    step = 0.1
    env = envelope(x, rate, step)
    end_s = len(x) / rate
    walk_s = float(walk_points_of(room).times[-1])
    first = min(quiet_moment(env, step, walk_s + LINGER_S, PAUSE_S), end_s)
    turned = turn_of(args)[-1][0]
    second = min(quiet_moment(env, step, first + turned + LINGER_S, PAUSE_S), end_s)
    if second - first < turned:
        what = guide_input(args)
        what = what.name if what is not None else f"the talk of {args.talker}"
        raise SystemExit(
            f"error: {what} is {end_s:.0f} s long, too short for the walk and the turn "
            f"({walk_s + turned + 2 * LINGER_S:.0f} s); pick another"
        )
    return first, second


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
    """The walk: responses every 25 cm along it, the late part traced every 2 m, the
    talk up to the first cut played through them; ``walk/walk.<format>.wav`` and
    walk.json (the places, the level and STI at each)."""
    from acoustic_spaces.spatial import Binaural
    from acoustic_spaces.walkthrough import render_walk, simulate_walk

    room = room_of(args)
    x, rate = talk(args)
    first, _ = cuts(args, room)
    print("simulating the walk (the late part takes the longest)")
    walked = simulate_walk(room, VISITOR, progress=print)
    rate_out = walked.result.settings.sample_rate
    ears = Binaural(hrtf_of(args, rate_out))
    wet = render_walk(walked, {"guide": (x[: int(first * rate)], rate)}, ears)
    out = args.work / "walk"
    write(out / f"walk.{label_of(args)}.wav", wet, rate_out)
    pipeline.write_json(
        out / WALK,
        {
            "talker": args.talker,
            "plays": played_by(args),
            "talk_s": [0.0, first],
            "times": walked.points.times.tolist(),
            "positions": walked.points.positions.tolist(),
            "yaws": yaws_of(walked, room).tolist(),
            **facts_of(room, [walked.result["guide", name] for name in walked.names]),
        },
    )
    print(f"wrote {out}: {walked.duration_s:.1f} s of walking, the talk to {first:.1f} s")


def yaws_of(walked, room) -> np.ndarray:
    """The way the visitor faces at each place of the walk: its own yaw all along when it
    does not face forward."""
    if walked.points.yaws is not None:
        return walked.points.yaws
    return np.full(len(walked.points.times), visitor_of(room.scene).orientation.yaw)


def turn(args) -> None:
    """The turn: the visitor standing at ``--stand``, simulated there once, the talk
    from the first cut through its response in Ambisonics, turned against the head as
    ``--turn`` scripts it and decoded to the ears (the engine's head tracker);
    ``turn/turn.<label>.wav`` and turn.json."""
    from acoustic_spaces import Room, simulate
    from acoustic_spaces.headtracking import HeadTrackedBinaural
    from acoustic_spaces.render import resample
    from acoustic_spaces.spatial import Ambisonics

    room = room_of(args)
    x, rate = talk(args)
    first, second = cuts(args, room)
    standing = [lis for lis in room.scene.listeners if lis.name == "standing"]
    if not standing:
        raise SystemExit(f"error: {args.scene} has no listener 'standing'; run: scene")
    alone = Room.from_scene(room.scene.model_copy(update={"listeners": standing}), reuse=room)
    print("simulating the place the visitor looks round from")
    result = simulate(alone, progress=print)
    response = result["guide", "standing"]
    rate_out = result.settings.sample_rate
    seed = result.settings.seed if result.settings.seed is not None else response.seed
    field = response.spatial_impulse_response(Ambisonics(TURN_ORDER), rate_out, seed)
    tracker = HeadTrackedBinaural(field, rate_out, hrtf=hrtf_of(args, rate_out))
    keys = turn_of(args)
    times, yaws = [t for t, _ in keys], [yaw for _, yaw in keys]
    dry = resample(x[int(first * rate) : int(second * rate)], rate, rate_out)
    print(f"turning the head through {len(keys)} keys")
    wet = tracker.run(dry, head=lambda t: (float(np.interp(t, times, yaws)), 0.0, 0.0))
    out = args.work / "turn"
    write(out / f"turn.{label_of(args)}.wav", wet, rate_out)
    facing = standing[0].orientation.yaw
    pipeline.write_json(
        out / TURNING,
        {
            "talker": args.talker,
            "plays": played_by(args),
            "talk_s": [first, second],
            "facing": facing,
            "keys": [list(key) for key in keys],
            "times": times,
            "positions": [list(standing[0].position)] * len(keys),
            "yaws": [facing + yaw for yaw in yaws],
            **facts_of(room, [response] * len(keys)),
        },
    )
    print(f"wrote {out}: the talk to {second:.1f} s")


# --------------------------------------------------------------------------- the video


def part_audio(path: Path, talk_s: float) -> tuple[np.ndarray, int]:
    """A part's sound to its talk's end and TAIL_S of the room after it, faded out."""
    x, rate = read(path)
    x = x[: int((talk_s + TAIL_S) * rate)]
    fade = int(FADE_S * rate)
    x[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None] ** 2
    return x, rate


def video(args) -> None:
    """tour.wav (the walk, a breath of silence, the turn: one gain, the loudest peak at
    -1 dBFS), and tour.mp4: the gallery from above, the visitor walking and turning, and
    what they hear, with a scrolling spectrogram (ffmpeg); stills of its moments."""
    from acoustic_spaces.demos.pictures import save, settings_lines

    pictures = sibling(__file__, "walk_pictures")
    label = label_of(args)
    parts = {}
    for name, info in (("walk", WALK), ("turn", TURNING)):
        path = args.work / name / f"{name}.{label}.wav"
        if not path.exists():
            raise SystemExit(f"error: no {path}; run first: {DEMO.command(name, args)}")
        facts = pipeline.read_json(args.work / name / info)
        start, stop = facts["talk_s"]
        parts[name] = (*part_audio(path, stop - start), facts)
    (walk_x, rate, walked), (turn_x, _, turned) = parts["walk"], parts["turn"]
    walk_x, turn_x = (np.atleast_2d(x.T).T for x in (walk_x, turn_x))
    gap = np.zeros((int(GAP_S * rate), walk_x.shape[1]))
    audio = np.concatenate([walk_x, gap, turn_x])
    audio *= 10.0 ** (-1.0 / 20.0) / float(np.abs(audio).max())
    write(args.work / "tour.wav", audio, rate)
    print(f"wrote {args.work / 'tour.wav'} ({len(audio) / rate:.1f} s)")
    room = room_of(args)
    scene_data = yaml.safe_load(args.scene.read_text(encoding="utf-8"))
    drawing = pictures.Pictures(
        scene_data,
        walked,
        turned,
        walk_s=len(walk_x) / rate,
        turn_at=(len(walk_x) + len(gap)) / rate,
        voice=(args.voice, VOICES[args.voice]),
        credit=credit_of(args),
        hrtf=args.hrtf,
        pace=pace_of(visitor_of(room.scene)),
        tool=settings_lines(DEMO, args, scene_data),
    )
    save(drawing.thumbnail(), args.work / "thumbnail.png")
    moments = drawing.moments()
    pipeline.animation(
        args.work,
        len(audio) / rate,
        drawing.frame,
        args.work / "tour.wav",
        stills=[(name, t + 0.5) for name, t, _ in moments],
        no_mp4=args.no_mp4,
        chapters=[f"{int(t // 60)}:{int(t % 60):02d} {title}" for _, t, title in moments],
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


def part_done(args, name: str, info: str) -> bool:
    """Whether the walk's or the turn's sound is there, made from what the guide says
    now (files written before they kept "plays" said the talk of their "talker") and
    from the scene as it is now."""
    path = args.work / name / info
    if not path.exists():
        return False
    if args.scene.exists() and path.stat().st_mtime < args.scene.stat().st_mtime:
        return False  # the scene changed since: the walk edited, say
    facts = pipeline.read_json(path)
    return facts.get("plays", f"ACE {facts.get('talker')}") == played_by(args)


def scene_done(args) -> bool:
    """Whether scene_demo.yaml is there and newer than the example's scene it is made from
    (the visitor's walk edited in the editor makes it again)."""
    if not args.scene.exists():
        return False
    return not args.source.exists() or args.scene.stat().st_mtime >= args.source.stat().st_mtime


def moving(args) -> list[str]:
    """Who the demo plays moving: the visitor, when the scene gives it a walk that is on."""
    from acoustic_spaces.scene import Scene

    try:
        visitor = visitor_of(Scene.from_yaml(args.source))
    except (Exception, SystemExit):
        return []
    return [VISITOR] if visitor.walk is not None and visitor.walk.enabled else []


def problems(demo, args) -> str | None:
    """The visitor must walk: the example's scene gives it a walk that is on."""
    from acoustic_spaces.scene import Scene

    source = Path(demo.resolved(args).source)
    if not source.exists():
        return f"no {source}"
    try:
        scene = Scene.from_yaml(source)
    except Exception as exc:  # the scene's own message
        return f"{source.name}: {exc}"
    visitor = next((lis for lis in scene.listeners if lis.name == VISITOR), None)
    if visitor is None:
        return f"{source.name} has no listener {VISITOR!r} to walk"
    if visitor.walk is None:
        return (
            f"the {VISITOR} has no walk in {source.name}: give it one (walk:, or the "
            "editor's Movement panel)"
        )
    if not visitor.walk.enabled:
        return (
            f"the {VISITOR}'s walk is off in {source.name}: turn it on (Moves in the "
            "editor, or walk: enabled: true)"
        )
    return None


DEMO = Demo(
    name="walk",
    example="gallery",
    title="A walk through the gallery",
    description=__doc__,
    script=Path(__file__),
    steps=[
        Step("speech", speech, "download the talk (the corpus's 148 MB archive)", speech_done),
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
            lambda args: part_done(args, "walk", WALK),
        ),
        Step(
            "turn",
            turn,
            "simulate where the visitor stands and turn its head",
            lambda args: part_done(args, "turn", TURNING),
        ),
        Step("video", video, "tour.wav, tour.mp4 and the stills in the work folder"),
    ],
    options=[
        Option(
            "talker",
            "the ACE corpus's talker",
            "M4",
            "choice",
            TALKERS,
            ("speech", "walk", "turn", "video"),
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
        Option("scene", "the scene with the guide's voice", Path("scene_demo.yaml"), "path"),
        Option(
            "stand",
            "x,y where the visitor stands to look round (m)",
            f"{STAND[0]:g},{STAND[1]:g}",
            steps=("scene",),
        ),
        Option(
            "turn",
            "t:yaw,... the head's yaw (degrees, turning left, 0 facing the guide) at each time (s)",
            None,
            steps=("turn",),
        ),
        *[o for o in common_options() if o.name not in ("again", "format")],
        dataclasses.replace(
            input_option(("walk", "turn", "video")),
            help="a WAV file the guide says instead of the ACE talk, as guide=WAV: mixed "
            "down to mono, at least as long as the walk and the turn need; the captions' "
            "level and STI are the scene's voice (--voice), as the video's one gain plays "
            "any recording as loud",
        ),
    ],
    scene_from="scene",
    problems=problems,
    plays=plays,
    moving=moving,
)


__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
