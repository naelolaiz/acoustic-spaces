"""A demo of any scene: what its sources play, heard at each of its listeners, as a video.

Each source plays the WAV file its ``signal`` names (set on the source's page in the
editor, **Plays**, or in the scene file), or the one ``--input SOURCE=WAV`` gives it
instead, dry: mixed down to mono, or the channel its ``channel`` names (**Channel** on
its page): left, right, or both mixed down (mono). A source with a channel and no signal
of its own plays that channel of the scene's ``recording`` (**Recording** on the scene's
page): a file of yours, such as a band's stereo mix, or one of the free recordings
(:mod:`acoustic_spaces.recordings`), which the ``recording`` step downloads the first
time and keeps for every scene that names it. So the left and right loudspeakers of a
sound system play the two sides of one recording, and its subwoofers both mixed down.
A file mixed down is brought to one gated RMS; its left and right channels share one
gain that brings their mean square to it, so a recording's stereo balance stays; then
each source's sound power in the scene (``level_db``) sets how loud it is against the
others, as the simulation says and not as the recordings happen to be. Sources with
neither stay silent. ``--sources`` plays some of them alone, by name or by the scene's
group (``--sources PA`` takes ``PA/Mains`` and ``PA/Fills``), the others silent, from
the same feeds and simulation; ``--sources-play in-turn`` plays each name it gives (or,
without it, each top group of those that play, a source of no group by its name) in a
part of its own, one after another in one video, and ``--series`` orders the parts.
Every listener in use is heard in turn, in the scene's order (or only those
``--listeners`` names, in its order), for ``--seconds`` each while the piece runs on, or
as the scene's tour (or ``--tour``) has them, each stop for phrases of the recording,
seconds or to its end, outdoors in the scene's weather or in still air, with one gain
for all the stops and a crossfade between them, binaural through the HRTFs
of the scene's binaural output (or ``--hrtf``, or a spherical head model when neither
names any). The stills show the scene from above with the listener of the moment, the
level there, how long the sound takes to arrive and T30 at mid frequencies, where a
talker plays (a source with a speech spectrum) the STI of its speech there, and say how
the sound was made. The simulation keeps every listener in use, so another choice of
``--listeners`` needs no new one. ``--recording`` plays another free recording, or a
file of yours, in place of the scene's, and ``--vocal-effort`` has every talker talk
relaxed, normal, raised or loud (ISO 9921); a run that changes a talker's sound power
or speech spectrum so simulates the scene as it plays it (:func:`run_scene_file`).

Sources and listeners whose trajectory is on move along it while the piece plays, all on one
clock from its start (:func:`acoustic_spaces.walkthrough.simulate_movement`): every
listener is heard as it moves or as the sources move past it, and the video moves them
as they go, its captions giving the level at each moment; a recording shorter than what
moves plays again from its start. ``--moving`` picks the trajectories
of a run without changing the scene: ``--moving visitor,guide`` moves those two and
leaves every other mover standing where it starts, ``--moving none`` keeps them all
still, so one scene serves both; without it, a source a given ``--sources`` leaves out
stands where it starts, as it plays nothing (without ``--sources`` a silent source moves
as the scene has it). A trajectory with Doppler (a car going past) moves
alone: every listener hears each early path through its own delay as it changes, so the
pitch shifts as it passes, the late sound crossfaded from place to place
(:func:`acoustic_spaces.walkthrough.render_movement`), all from one simulation.

    acoustic-spaces demo scene.yaml              # every step, into demo/ beside the scene
    acoustic-spaces demo scene.yaml recording    # download the scene's free recording
    acoustic-spaces demo scene.yaml feeds        # one step
    acoustic-spaces demo scene.yaml --hrtf me.sofa --seconds 15 --listeners front,back
    acoustic-spaces demo scene.yaml --input talker=other_take.wav
    acoustic-spaces demo scene.yaml --moving none   # every mover standing where it starts
    acoustic-spaces demo scene.yaml --sources PA     # the sources of the group PA alone
    acoustic-spaces demo scene.yaml --sources PA,band --sources-play in-turn
    acoustic-spaces demo scene.yaml --recording ace_f1 --vocal-effort raised
    acoustic-spaces demo scene.yaml --recording song.flac --bpm 120 --tour front:2,back:end

The editor runs it from **Simulation > Run demo…** as *This scene*, with a tick for each
trajectory in its Movement table, for each listener in its Listeners list and for each source
in its Sources list. The video needs ffmpeg on the PATH and Pillow (the ``viz`` extra).
The demos of the examples are scripts of their own, with their recordings, the gallery's
walk a preset of this one (:func:`preset`); this one takes any scene.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import shutil
import sys
from pathlib import Path

import numpy as np

from acoustic_spaces import recordings
from acoustic_spaces.demos import (
    SIMULATE,
    Demo,
    Option,
    Source,
    Step,
    common_options,
    given_names,
    input_option,
    listener_names,
    listeners_in,
    listeners_option,
    mix_stem,
    moving_names,
    name_part,
    pipeline,
    playing,
    product,
    run_folder,
    run_plays,
    sources_options,
)
from acoustic_spaces.demos.audio import (
    channel_weights,
    channels_rms,
    decoded,
    gated_rms,
    level_db,
    loudness_db,
    mono,
    read,
    resampled,
    stereo,
    write,
)
from acoustic_spaces.demos.music import AUDIO_FILTER
from acoustic_spaces.i18n import N_
from acoustic_spaces.parameters import VOCAL_EFFORT_DB_A
from acoustic_spaces.scene import Scene, TourStop
from acoustic_spaces.scene.model import ATMOSPHERES, CHANNELS

#: What the feeds are brought to (gated RMS, full scale): the loudness every source's
#: signal has before the simulation's sound powers set the balance.
FEED_RMS = 0.1
#: The feeds, and what each source plays, in the work folder.
FEEDS = "feeds.json"
#: How the feeds are levelled, kept in FEEDS: feeds levelled otherwise (a stereo
#: recording's channels from their mixdown, before 4.1.0) are made again.
FEED_LEVELS = "channels gated together"
#: --hrtf here: a scene names its own HRTFs in its binaural output.
HRTF_HELP = (
    "binaural mixes through this SOFA file of HRTFs, your own or measured, instead of the "
    "one the scene's binaural output names (or a spherical head model when it names none)"
)
#: --sources here: every source in use that has something to play, by default.
SOURCES_HELP = (
    "only these sources, by name or by the scene's group (PA takes PA/Mains and "
    "PA/Fills), comma separated (default: every source in use that has something to "
    "play: its signal, an --input file or a channel of the scene's recording); the others "
    "stay silent, and as the feeds and the simulation keep every source, another pick "
    "needs neither again"
)
#: What a source plays: the file, and the channel of it (None: mixed down to mono), or
#: the recording and the stem of it the source plays, or the mix of its stems (stem to
#: gain, dB), those the recording has.
Played = tuple[Path, str | dict[str, float] | None]
#: How :func:`plays` names a channel.
CHANNEL_WORDS = {"left": "left channel", "right": "right channel", "mono": "mixed down to mono"}
CROSSFADE_S = 1.0
FADE_OUT_S = 3.0
#: The folder of the mixes when something moves (the still ones go to ``seats``).
MOVING = "moving"
#: How long a stop lasts without ``--seconds`` on a shared timeline, or from a
#: listener's start where nothing moves (s).
SECONDS = 20.0
#: The silence between two stops that each start from the clock's start (s).
GAP_S = 1.0
#: The silence before a recording shorter than what moves plays again from its start (s).
AGAIN_S = 2.0
#: What a listener's mix in still air is keyed by (``audience.still``), where the run
#: hears the scene's own atmosphere and a stop of its tour still air.
STILL = "still"
#: The run's timelines (``--timeline``): one clock the tour moves along from listener
#: to listener, or each listener heard from the clock's start; ``auto`` (the default),
#: the second when something moves and the first otherwise.
TIMELINES = ("auto", "shared", "per-listener")
#: The vocal efforts a run can give every talker (``--vocal-effort``, ISO 9921).
VOCAL_EFFORTS = tuple(VOCAL_EFFORT_DB_A)
#: The folder of the scenes as the runs that change them play them
#: (:func:`run_scene_file`), in the work folder.
SCENES = "scenes"


def work_of(args: argparse.Namespace) -> Path:
    """Where the demo writes: ``--work``, or ``demo`` beside the scene."""
    return Path(args.work) if args.work else Path(args.scene).parent / "demo"


def results_at(args: argparse.Namespace, scene: str | Path) -> Path:
    """Where the simulation of ``scene`` for the output a run asks for is kept, in the
    work folder: ``results/<the scene's name>/<the output>``
    (:func:`~acoustic_spaces.demos.pipeline.results_for`), so that scenes sharing a work
    folder, and runs of other outputs, simulate over none of each other's."""
    return pipeline.results_for(work_of(args) / "results" / name_part(Path(scene).stem), args)


def feeds_folder(args: argparse.Namespace) -> Path:
    """Where a run's feeds go: ``feeds/<scene>`` in the work folder, or with the files
    ``--input`` gives the sources, ``feeds/<scene>_<what plays>``
    (:func:`~acoustic_spaces.demos.run_plays`): what a run feeds hangs on nothing else,
    so that the runs that play the same share them."""
    plays = run_plays(args)
    name = name_part(stem_of(args)) + (f"_{plays}" if plays else "")
    return work_of(args) / "feeds" / name


def stem_of(args: argparse.Namespace) -> str:
    """The first part of a run's name: the scene file's name."""
    return Path(args.scene).stem if getattr(args, "scene", None) else "scene"


@functools.lru_cache(maxsize=8)
def _scene_at(path: str, _changed: int) -> Scene | None:
    """The scene at ``path`` as it is now (``_changed``, its time of change, keeps the
    cache fresh), or None when it cannot be read: :func:`problems` says why."""
    try:
        return Scene.from_yaml(path)
    except Exception:
        return None


def hidden(args: argparse.Namespace) -> list[str]:
    """What ``--hide`` leaves out of the stills, by name."""
    given = getattr(args, "hide", None) or ""
    return [part.strip() for part in str(given).split(",") if part.strip()]


def scene_pictures_hidden() -> tuple[str, ...]:
    """What the stills can leave out (``--hide``)."""
    from acoustic_spaces.demos.scene_pictures import HIDDEN

    return HIDDEN


def _current(scene: str | Path | None) -> Scene | None:
    if not scene or not Path(scene).exists():
        return None
    return _scene_at(str(Path(scene).resolve()), Path(scene).stat().st_mtime_ns)


def plays_recording(source, inputs: dict | None = None) -> bool:
    """Whether ``source`` (of a scene) plays the scene's recording: it has a channel,
    and neither a signal nor a file ``inputs`` (``--input``) gives it."""
    return (
        source.channel is not None and source.signal is None and source.name not in (inputs or {})
    )


def stand_of(text: str) -> tuple[str, float, float]:
    """``--stand``'s listener and where it stands, from ``LISTENER=X,Y`` (m); ValueError
    for another form."""
    name, sep, place = str(text).partition("=")
    try:
        x, y = (float(v) for v in place.split(","))
    except ValueError:
        sep = ""
    if not sep or not name.strip():
        raise ValueError(f"--stand takes LISTENER=X,Y in metres, not {text!r}")
    return name.strip(), x, y


def turn_of(text: str) -> tuple[str, list[tuple[float, float]]]:
    """``--turn``'s listener and its turn, from ``LISTENER=T:YAW,T:YAW,...``: the yaw
    (degrees, turning left from the way it faces) at each time (s), the times rising
    from 0; ValueError for another form."""
    name, sep, keys = str(text).partition("=")
    turn: list[tuple[float, float]] = []
    try:
        for key in keys.split(","):
            t, yaw = key.split(":")
            turn.append((float(t), float(yaw)))
    except ValueError:
        sep = ""
    times = [key[0] for key in turn]
    if not sep or not name.strip() or times[0] < 0.0 or sorted(set(times)) != times:
        raise ValueError(
            f"--turn takes LISTENER=T:YAW,... with the times rising from 0, not {text!r}"
        )
    return name.strip(), turn


def stood(scene: Scene, name: str, x: float, y: float) -> Scene:
    """``scene`` with the listener ``name`` standing at ``x``, ``y`` (its height kept) and
    facing the nearest source in use, its trajectory moved and turned with it, so that a
    turn on the spot stays one."""
    from acoustic_spaces.scene.model import Orientation

    listener = next(lis for lis in scene.listeners if lis.name == name)
    near = min(
        (s for s in scene.sources if s.in_use),
        key=lambda s: math.dist(s.position[:2], (x, y)),
        default=None,
    )
    yaw = listener.orientation.yaw
    if near is not None:
        yaw = round(math.degrees(math.atan2(near.position[1] - y, near.position[0] - x)), 2)
    shift = np.array([x, y, listener.position[2]]) - np.asarray(listener.position)
    turned = yaw - listener.orientation.yaw

    def moved(point) -> tuple[float, float, float]:
        a, b, c = (float(v) for v in np.asarray(point) + shift)
        return a, b, c

    def faced(orientation):
        if orientation is None:
            return None
        return orientation.model_copy(update={"yaw": orientation.yaw + turned})

    trajectory = listener.trajectory
    if trajectory is not None:
        keyframes = trajectory.keyframes
        if keyframes is not None:
            keyframes = [
                frame.model_copy(
                    update={
                        "position": moved(frame.position),
                        "orientation": faced(frame.orientation),
                    }
                )
                for frame in keyframes
            ]
        trajectory = trajectory.model_copy(
            update={"to": [moved(point) for point in trajectory.to], "keyframes": keyframes}
        )
    own = listener.orientation
    listener = listener.model_copy(
        update={
            "position": moved(listener.position),
            "orientation": Orientation(yaw=yaw, pitch=own.pitch, roll=own.roll),
            "trajectory": trajectory,
        }
    )
    return _with_listener(scene, listener)


def turning(scene: Scene, name: str, turn: list[tuple[float, float]]) -> Scene:
    """``scene`` with the listener ``name`` turning on the spot as ``turn`` (time s, yaw
    degrees turning left from the way it faces) has it, in place of its trajectory: a
    keyframe at each time after 0 s, the yaw at 0 s its own."""
    from acoustic_spaces.scene.model import Keyframe, Orientation, Trajectory

    listener = next(lis for lis in scene.listeners if lis.name == name)
    facing = listener.orientation
    if turn[0][0] == 0.0:
        listener = listener.model_copy(
            update={"orientation": facing.model_copy(update={"yaw": facing.yaw + turn[0][1]})}
        )
    frames = [
        Keyframe(
            time_s=t,
            position=listener.position,
            orientation=Orientation(yaw=facing.yaw + yaw, pitch=facing.pitch, roll=facing.roll),
        )
        for t, yaw in turn
        if t > 0.0
    ]
    trajectory = Trajectory(keyframes=frames) if frames else None
    return _with_listener(scene, listener.model_copy(update={"trajectory": trajectory}))


def still_air(scene: Scene) -> Scene:
    """``scene`` in still air: without its atmosphere (wind, temperature profiles, weather
    zones, turbulence) and its long-term weather classes, so that sound goes straight;
    its tour's stops in still air in its only air."""
    tour = scene.tour and [stop.model_copy(update={"atmosphere": "scene"}) for stop in scene.tour]
    return scene.model_copy(update={"atmosphere": None, "long_term": None, "tour": tour})


def _with_listener(scene: Scene, listener) -> Scene:
    """``scene`` with ``listener`` in place of its listener of that name."""
    listeners = [listener if lis.name == listener.name else lis for lis in scene.listeners]
    return scene.model_copy(update={"listeners": listeners})


def changed(args: argparse.Namespace, scene: Scene) -> Scene:
    """``scene`` as this run plays it: with ``--stand``, a listener standing elsewhere
    (:func:`stood`), and with ``--turn``, turning on the spot (:func:`turning`); with
    ``--recording``, that recording in place of its own, the sources that play it with
    a speech spectrum taking its talker's (a woman's or a man's); with
    ``--vocal-effort``, every talker in use (a source with a speech spectrum) talking so
    (:meth:`~acoustic_spaces.scene.Source.talking`); with ``--atmosphere still``, in
    still air (:func:`still_air`). ValueError for a ``--stand`` or a ``--turn`` that does
    not read, or names a listener the scene does not have."""
    names = [lis.name for lis in scene.listeners]
    stand, turn = getattr(args, "stand", None), getattr(args, "turn", None)
    if stand is not None:
        name, x, y = stand_of(stand)
        if name not in names:
            raise ValueError(f"--stand: no listener {name!r}; the scene has {', '.join(names)}")
        scene = stood(scene, name, x, y)
    if turn is not None:
        name, keys = turn_of(turn)
        if name not in names:
            raise ValueError(f"--turn: no listener {name!r}; the scene has {', '.join(names)}")
        scene = turning(scene, name, keys)
    if getattr(args, "atmosphere", None) == STILL and not scene.still:
        scene = still_air(scene)
    name = getattr(args, "recording", None)
    effort = getattr(args, "vocal_effort", None)
    if name is None and effort is None:
        return scene
    free = recordings.named(name) if isinstance(name, str) else None  # else a file of yours
    inputs = getattr(args, "inputs", None) or {}
    sources = []
    for source in scene.sources:
        talks = source.spectrum != "flat"
        if free is not None and free.talker and talks and plays_recording(source, inputs):
            spectrum = f"speech_{free.talker}"
            if spectrum != source.spectrum:
                source = source.model_copy(update={"spectrum": spectrum})
                source = source.talking(source.vocal_effort)  # its level, of that speech
        if effort is not None and talks and source.in_use:
            source = source.talking(effort)
        sources.append(source)
    update: dict = {"sources": sources}
    if name is not None:
        update["recording"] = free.name if free is not None else Path(name)
    return scene.model_copy(update=update)


def _run(args: argparse.Namespace) -> Scene | None:
    """The scene as this run plays it (:func:`changed`) as its file is now, or None when
    it cannot be read or the run's changes do not read: :func:`problems` says why."""
    found = _current(getattr(args, "scene", None))
    try:
        return None if found is None else changed(args, found)
    except ValueError:
        return None


def talkers_of(args: argparse.Namespace) -> list[str]:
    """The talkers in use of the scene, by name: its sources with a speech spectrum."""
    found = _current(getattr(args, "scene", None))
    if found is None:
        return []
    return [s.name for s in found.sources if s.in_use and s.spectrum != "flat"]


def stems_moot(args: argparse.Namespace) -> str | None:
    """Why ``--stem`` changes nothing in the run, or None: the run plays a free song (it
    has its own stems), no recording, or no source of it plays a stem."""
    scene = _run(args)
    if scene is None:
        return None
    free = free_of(scene)
    if free is not None:
        return f"the run plays {free.name}, a free recording: --stem is for one of yours"
    if scene.recording is None:
        return "the run plays no recording: give one of yours (--recording)"
    inputs = getattr(args, "inputs", None) or {}
    if not any(s.stems for s in scene.sources if s.in_use and plays_recording(s, inputs)):
        return "no source in use plays a stem: give one a stem or a mix of them as its channel"
    return None


def recording_heard(args: argparse.Namespace) -> bool:
    """Whether a source in use plays the scene's recording (:func:`plays_recording`)."""
    found = _current(getattr(args, "scene", None))
    inputs = getattr(args, "inputs", None)
    return found is not None and any(s.in_use and plays_recording(s, inputs) for s in found.sources)


def run_scene_file(args: argparse.Namespace) -> Path:
    """The scene file the run simulates: the scene's own, or, where the run changes what
    its simulation hears (still air, a talker's level or speech, where a listener stands
    or how it faces: :func:`changed`), the scene so
    changed, written to ``scenes/<the scene's name>_<the changes>.yaml`` in the work
    folder, again only when it changes, so that its simulation, once made, stays valid
    and a run back to the scene's own finds that one."""
    path = Path(args.scene)
    found = _current(path)
    run = _run(args)
    if found is None or run is None:
        return path
    before = {s.name: s for s in found.sources}
    levelled = [s for s in run.sources if not np.allclose(s.power_db(), before[s.name].power_db())]
    placed = {lis.name: (lis.position, lis.orientation) for lis in found.listeners}
    moved = [lis for lis in run.listeners if (lis.position, lis.orientation) != placed[lis.name]]
    still = run.still and not found.still
    if not levelled and not moved and not still:
        return path
    effort = getattr(args, "vocal_effort", None)
    parts = ["still-air"] if still else []
    parts += [] if effort is None or not levelled else [f"vocal-effort-{effort}"]
    spoken = sorted({s.talker for s in levelled if s.spectrum != before[s.name].spectrum})
    parts += [f"{talker}-speech" for talker in spoken]
    parts += [name_part(f"{lis.name}-at-{lis.position[0]:g}-{lis.position[1]:g}") for lis in moved]
    out = work_of(args) / SCENES / f"{name_part(path.stem)}_{'_'.join(parts)}{path.suffix}"
    text = (
        f"# {path.name} as a demo run plays it ({', '.join(parts)}), written by the demo:\n"
        "# change the scene itself, not this.\n" + run.file_text(out.parent, readable=True)
    )
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return out


def _scene_hrtf(scene: str | Path | None) -> Path | None:
    """The HRTFs the binaural output of the scene names, or None."""
    found = _current(scene)
    return None if found is None else pipeline.scene_hrtf(found.outputs)


def recording_of(scene: Scene) -> Path | None:
    """The file the scene's ``recording`` is: its own, or where the free one it names is
    kept (downloaded or not yet); None when it names none."""
    if scene.recording is None:
        return None
    if isinstance(scene.recording, Path):
        return scene.recording
    return recordings.file_of(scene.recording)


def free_of(scene: Scene) -> recordings.Recording | None:
    """The free recording the scene names, or None."""
    return recordings.named(scene.recording) if isinstance(scene.recording, str) else None


def stems_of(args: argparse.Namespace, scene: Scene) -> dict[str, Path]:
    """The stems of the recording the run plays and their files: a free song's (there or
    still to download, :func:`acoustic_spaces.recordings.stem_file`), or those
    ``--stem`` gives a recording of yours; none for a recording without."""
    free = free_of(scene)
    if free is not None:
        return {name: recordings.stem_file(free, name) for name in free.stem_names}
    if scene.recording is None:
        return {}
    return {name: Path(file) for name, file in (getattr(args, "stems", None) or {}).items()}


def signals_of(args: argparse.Namespace, scene: Scene) -> dict[str, Played | None]:
    """What each source in use plays, and which channel of it: the file ``--input``
    gives it, else its signal (the channel left, right or mono of it), else, with a
    channel, the scene's recording: a channel of it, or the stem the channel names or
    the mix of them, those the recording has (:func:`stems_of`). None: silent, as is a
    source none of whose stems the recording has."""
    inputs = getattr(args, "inputs", None) or {}
    recording = recording_of(scene)
    stems = stems_of(args, scene)
    out: dict[str, Played | None] = {}
    for source in scene.sources:
        if not source.in_use:
            continue
        file = inputs.get(source.name) or source.signal
        if file is not None:
            channel = source.channel if source.channel in CHANNELS else None
            out[source.name] = (Path(file), channel)
        elif source.channel is None or recording is None:
            out[source.name] = None
        elif not source.stems:
            out[source.name] = (recording, source.channel)
        else:
            have = {name: gain for name, gain in source.stems.items() if name in stems}
            if not have:
                out[source.name] = None
            elif isinstance(source.channel, str):
                out[source.name] = (recording, source.channel)
            else:
                out[source.name] = (recording, have)
    return out


def stems_played(played: Played | None) -> dict[str, float]:
    """The stems ``played`` plays and their gains (dB): none for a file or a channel."""
    if played is None or played[1] is None:
        return {}
    channel = played[1]
    if isinstance(channel, dict):
        return dict(channel)
    return {} if channel in CHANNELS else {channel: 0.0}


def channel_words(channel: str | dict[str, float] | None) -> str:
    """How :func:`plays` and the feeds name what a source plays of its file: a channel,
    a stem (``its vocal stem``) or a mix (``a mix of its vocal, backing -6 dB ...``)."""
    if channel is None:
        return ""
    if isinstance(channel, dict):
        parts = [name if gain == 0 else f"{name} {gain:+g} dB" for name, gain in channel.items()]
        return f"a mix of its stems {', '.join(parts)}"
    if channel in CHANNEL_WORDS:
        return CHANNEL_WORDS[channel]
    return f"its {channel} stem"


def plays(args: argparse.Namespace) -> dict[str, str]:
    """Every source in use and what it plays unless ``--input`` gives it a file: the name
    of its signal or of the scene's recording, and the channel, or silent."""
    found = _run(args)
    if found is None:
        return {}
    free = free_of(found)
    free_file = recordings.file_of(free) if free is not None else None
    out = {}
    own = argparse.Namespace(**{**vars(args), "inputs": None})
    sources = {source.name: source for source in found.sources}
    for name, played in signals_of(own, found).items():
        if played is None:
            wanted = sources[name].stems if sources[name].signal is None else {}
            heard = recording_of(found)
            if wanted and heard is not None and free is not None:
                out[name] = f"silent: {free.title} has no {', '.join(wanted)} stem"
            elif wanted and heard is not None:
                out[name] = f"silent: no {', '.join(wanted)} stem of {heard.name} (--stem)"
            else:
                out[name] = "silent"
            continue
        file, channel = played
        what = free.title if free is not None and file == free_file else file.name
        out[name] = what if channel is None else f"{what}, {channel_words(channel)}"
    return out


def sources_of(args: argparse.Namespace) -> list[Source]:
    """The sources a run can play (``--sources``): every source in use, in the scene's
    order, with its group; those with something to play (:func:`signals_of`) play unless
    ``--sources`` leaves them out. None at all when the scene cannot be read
    (:func:`problems` says why)."""
    found = _run(args)
    if found is None:
        return []
    groups = {source.name: source.group for source in found.sources}
    return [
        Source(name, groups[name], played is not None)
        for name, played in signals_of(args, found).items()
    ]


def downloads_of(args: argparse.Namespace) -> tuple[recordings.Recording | None, bool, bool]:
    """The free recording the run plays, whether it is still to download (a source plays
    it, or its stems, which are lined up with it), and whether its stems are (a source
    plays one)."""
    scene = _run(args)
    free = free_of(scene) if scene is not None else None
    if free is None:
        return None, False, False
    played = [v for v in signals_of(args, scene).values() if v is not None]
    stems = any(stems_played(v) for v in played) and not recordings.stems_there(free)
    heard = any(v[0] == recordings.file_of(free) for v in played)
    return free, heard and not recordings.file_of(free).exists(), stems


def recording_needed(args: argparse.Namespace) -> recordings.Recording | None:
    """The free recording the run plays whose file or stems are not downloaded yet, or
    None."""
    free, master, stems = downloads_of(args)
    return free if master or stems else None


def recording_download(args: argparse.Namespace) -> str | None:
    """What ``recording`` downloads: the free recording and its stems, as the run needs
    them, its licence and size."""
    free, master, stems = downloads_of(args)
    if free is None or not (master or stems):
        return None
    parts = [f"{free.title} ({free.licence}, {free.download_mb:g} MB)"] if master else []
    if stems:
        whose = "its" if master else f"of {free.title}"
        parts.append(
            f"{whose} {len(free.stems)} stems ({free.stems_mb:g} MB)"
            if master
            else f"the {len(free.stems)} stems {whose} ({free.stems_mb:g} MB)"
        )
    return f"{' and '.join(parts)}, kept for every scene"


def recording(args: argparse.Namespace) -> None:
    """The free recording the scene names, downloaded into the recordings folder
    (:func:`acoustic_spaces.recordings.data_folder`) unless it is there or nothing plays
    it, and its stems where a source plays one."""
    free, master, stems = downloads_of(args)
    if free is None or not (master or stems):
        scene = _run(args)
        named = free_of(scene) if scene is not None else None
        if named is None:
            print("the scene names no free recording: nothing to download")
        else:
            print(f"{named.title}: there already, or nothing plays it")
        return
    print(f"{free.title}: {recordings.get(free)}")
    if stems:
        print(f"its stems: {recordings.stems_folder(free)}")
        recordings.get_stems(free)


def completed(args: argparse.Namespace) -> argparse.Namespace:
    """The arguments with the scene's own HRTFs when ``--hrtf`` names none and the
    mixes are binaural: a scene whose binaural output names a SOFA file is heard through
    it, as its simulation is."""
    if getattr(args, "hrtf", None) or getattr(args, "format", "binaural") != "binaural":
        return args
    hrtf = _scene_hrtf(getattr(args, "scene", None))
    if hrtf is not None:
        args.hrtf = hrtf
    return args


def run_movers(args: argparse.Namespace, scene: Scene) -> list[str] | None:
    """Who moves in this run, as :func:`~acoustic_spaces.walkthrough.with_moving` takes
    it: those ``--moving`` names; without it, when ``--sources`` is given and leaves out
    a source whose trajectory is on, every mover of the scene but that one, which stands where
    it starts, as it plays nothing; None: the scene's own movers (without ``--sources``
    too, a source the scene keeps silent moving as the scene has it)."""
    from acoustic_spaces.walkthrough import movers

    names = moving_names(getattr(args, "moving", None))
    chosen = getattr(args, "playing", None)
    if names is not None or chosen is None or given_names(getattr(args, "sources", None)) is None:
        return names
    movers = movers(scene)
    kept = [item.name for kind, item in movers if kind == "listeners" or item.name in chosen]
    return None if len(kept) == len(movers) else kept


def moving_scene(args: argparse.Namespace) -> Scene:
    """The scene as this run plays it: with ``--moving``, those move and every other
    mover stands where it starts (:func:`~acoustic_spaces.walkthrough.with_moving`); a
    source a given ``--sources`` leaves out stands too (:func:`run_movers`); every
    trajectory that moves at ``--speed`` and simulated every ``--step``, when given
    (:func:`run_trajectories`)."""
    from acoustic_spaces.walkthrough import with_moving

    found = scene_of(args)
    return run_trajectories(args, with_moving(found, run_movers(args, found)))


def run_trajectories(args: argparse.Namespace, scene: Scene) -> Scene:
    """``scene`` with each trajectory that moves at the run's ``--speed`` (km/h; along
    keyframes, through their places at that speed:
    :func:`~acoustic_spaces.walkthrough.at_speed`) and its places ``--step`` apart (m),
    where the run gives them; as it is otherwise."""
    from acoustic_spaces.walkthrough import at_speed, movers

    speed = getattr(args, "speed", None)
    step = getattr(args, "step_m", None)
    if speed is None and step is None:
        return scene
    moving = {(kind, item.name) for kind, item in movers(scene)}
    update = {}
    for kind in ("listeners", "sources"):
        items = []
        for item in getattr(scene, kind):
            if (kind, item.name) in moving:
                if speed is not None:
                    item = at_speed(item, float(speed) / 3.6)
                if step is not None:
                    trajectory = item.trajectory.model_copy(update={"step_m": float(step)})
                    item = item.model_copy(update={"trajectory": trajectory})
            items.append(item)
        update[kind] = items
    return scene.model_copy(update=update)


def timeline_of(args: argparse.Namespace, moving: list[str]) -> str:
    """The run's timeline, ``shared`` or ``per-listener``: as ``--timeline`` says, or with
    ``auto`` (the default) each listener from the clock's start when something moves
    (``moving``), and one shared clock otherwise."""
    chosen = getattr(args, "timeline", None) or "auto"
    if chosen != "auto":
        return str(chosen)
    return "per-listener" if moving else "shared"


def moving_of(args: argparse.Namespace) -> list[str]:
    """Who moves in this run (:func:`run_movers`): the sources and listeners in use whose
    trajectories are on in the scene as the run plays it (:func:`changed`: a ``--turn``
    turns one), by name, listeners first; none when the scene cannot be read or
    ``--moving`` names what does not move (:func:`problems` says why)."""
    from acoustic_spaces.walkthrough import movers, with_moving

    scene = _run(args)
    if scene is None:
        return []
    try:
        scene = with_moving(scene, run_movers(args, scene))
    except ValueError:
        return []
    return [item.name for _kind, item in movers(scene)]


def results_of(args: argparse.Namespace) -> Path:
    """Where the simulation of the scene as the run plays it (:func:`run_scene_file`) is
    kept for the run's output (:func:`results_at`)."""
    return results_at(args, run_scene_file(args))


def scene_of(args: argparse.Namespace) -> Scene:
    """The scene as the run plays it (:func:`changed`), its sources' signals resolved
    against its folder."""
    return changed(args, Scene.from_yaml(args.scene))


def simulations(args: argparse.Namespace) -> list[pipeline.Simulation]:
    """The simulation of the scene as the run plays it (:func:`run_scene_file`), and of
    the same in still air when a stop of its tour is heard so (:func:`still_stops`)."""
    paths = [run_scene_file(args)]
    if still_stops(args):
        paths.append(run_scene_file(still_args(args)))
    return [pipeline.Simulation(path, results_at(args, path), args.format) for path in paths]


def responses_of(args: argparse.Namespace, demo: Demo | None = None) -> list[dict]:
    """The responses of the scene's simulation; ``demo`` (this one by default) says how
    to run it when they are not there."""
    run = (demo or DEMO).command("simulate", args)
    return pipeline.responses_of(simulations(args)[0], args, run)


def problems(demo: Demo, args: argparse.Namespace) -> str | None:
    """A scene is needed, it must be there, and a source of it must play a WAV file
    that is there (its signal, or the one ``--input`` gives it); one of those
    ``--sources`` picks must play something."""
    if not getattr(args, "scene", None):
        return "give the scene to demo (--scene)"
    for name in ("seconds", "speed", "step_m", "bpm"):
        value = getattr(args, name, None)
        if value is not None and not float(value) > 0.0:
            return f"{demo.option(name).option} is a positive number, not {value!r}"
    if getattr(args, "downbeat", None) is not None and getattr(args, "bpm", None) is None:
        return "--downbeat is for a recording of yours, with its tempo (--bpm)"
    last = getattr(args, "last_stop", None)
    if last:
        try:
            length_of_stop(last)
        except ValueError as exc:
            return f"--last-stop: {exc}: give phrases (2), seconds (12s) or end"
    unknown = [part for part in hidden(args) if part not in scene_pictures_hidden()]
    if unknown:
        return f"--hide {unknown[0]}: give one or more of {', '.join(scene_pictures_hidden())}"
    resolved = demo.resolved(args)
    path = Path(resolved.scene)
    if not path.exists():
        return f"no {path}"
    try:
        scene = Scene.from_yaml(path)
    except Exception as exc:  # the scene's own message
        return f"{path.name}: {exc}"
    try:
        scene = changed(resolved, scene)
    except ValueError as exc:
        return str(exc)
    free = free_of(scene)
    if getattr(args, "bpm", None) is not None and free is not None and free.bpm is not None:
        return f"--bpm is for a recording of yours: {free.name} has its own, {free.bpm:g} bpm"
    signals = {k: v for k, v in signals_of(resolved, scene).items() if v is not None}
    if not signals:
        return (
            "no source plays anything: give one a WAV file (Plays on its page in the "
            "editor, signal: in the scene file, or --input SOURCE=WAV), or give the scene "
            "a recording and the sources that play it a channel"
        )
    chosen = getattr(resolved, "playing", None)
    if chosen and not any(name in chosen for name in signals):
        return (
            f"--sources picks only sources that play nothing ({', '.join(chosen)}); "
            f"those that play: {', '.join(signals)}"
        )
    downloaded = recordings.file_of(free) if free is not None else None
    for name, (signal, _channel) in signals.items():
        if signal == downloaded:  # the recording step downloads it
            continue
        if not signal.exists():
            return f"{name} plays {signal}, which is not there"
        if signal.suffix.lower() != ".wav" and shutil.which("ffmpeg") is None:
            return (
                f"{name} plays {signal.name}: ffmpeg is needed to decode it, and is not on the PATH"
            )
    from acoustic_spaces.walkthrough import movers, with_moving

    try:
        moving = with_moving(scene, run_movers(resolved, scene))
    except ValueError as exc:
        return f"--moving: {exc}"
    movers = [item for _kind, item in movers(moving)]
    for item in movers:
        if item.trajectory.doppler and len(movers) > 1:
            return (
                f"{item.name} moves with Doppler, which plays alone: leave the others "
                f"standing (--moving {item.name}), or turn its Doppler off"
            )
    return tour_problem(resolved, scene, [item.name for item in movers])


def tour_problem(args: argparse.Namespace, scene: Scene, moving: list[str]) -> str | None:
    """What is wrong with the run's tour (:func:`tour_given`), ``scene`` as the run plays
    it and ``moving`` what moves in it: a ``--tour`` that does not read, a stop at a
    listener the run does not hear, in still air where the scene's is still already or
    where something moves, in phrases where the recording has no tempo, or a tour where
    each listener is heard from the clock's start."""
    given = getattr(args, "tour", None)
    try:
        stops = tour_given(args, scene)
    except ValueError as exc:
        return str(exc)
    if stops is None:
        return None
    if not stops:
        return f"--tour stops at none of the listeners --listeners names ({args.listeners})"
    shared = timeline_of(args, moving) == "shared"
    if not shared:
        return (
            "--tour: each listener is heard from the clock's start where something moves; "
            "a tour plays on a shared timeline (--timeline shared)"
            if given
            else None
        )
    own = _current(getattr(args, "scene", None))
    if given and own is not None and own.still:  # tour_given hears it in the only air
        still = next((s for s in tour_text(given) if s.atmosphere == STILL), None)
        if still is not None:
            return (
                f"--tour hears {still.listener!r} in still air, where the scene's air is "
                "still already"
            )
    heard = [lis.name for lis in scene.listeners if lis.in_use]
    for stop in stops:
        if stop.listener not in heard:
            return f"the tour stops at {stop.listener!r}, which is not in use; the run hears " + (
                ", ".join(heard)
            )
        if stop.atmosphere == STILL and moving:
            return (
                f"the tour hears {stop.listener!r} in still air, while {moving[0]} moves: a "
                "stop in still air is for a run where nothing moves (--moving none)"
            )
    if tempo_of(args, scene) is None:
        phrased = next((s for s in stops if s.phrases is not None), None)
        if phrased is not None:
            return (
                f"the tour's stop at {phrased.listener} lasts {phrased.phrases:g} phrases, and "
                "the recording has no tempo: give it seconds (12s), or the tempo (--bpm)"
            )
    return None


# --------------------------------------------------------------------------- the feeds


def feeds_table(args: argparse.Namespace, demo: Demo | None = None) -> dict:
    path = feeds_folder(args) / FEEDS
    if not path.exists():
        raise SystemExit(f"error: no feeds yet; run first: {(demo or DEMO).command('feeds', args)}")
    return pipeline.read_json(path)


def feeds_done(args: argparse.Namespace) -> bool:
    """Whether the feeds are there, play what the scene and ``--input`` ask for, and are
    newer than the scene and the files they play."""
    path = feeds_folder(args) / FEEDS
    if not path.exists():
        return False
    made = path.stat().st_mtime
    table = pipeline.read_json(path)
    if table.get("levels") != FEED_LEVELS:
        return False
    scene = _run(args)
    if scene is None:
        return False
    wanted = feed_entries(args, scene)
    kept = ("signal", "channel", "stems")
    if wanted != {k: {f: v.get(f) for f in kept} for k, v in table["sources"].items()}:
        return False
    files = [Path(args.scene)]
    for entry in wanted.values():
        files += [Path(entry["signal"]), *(Path(f) for f in (entry["stems"] or {}).values())]
    return all(not f.exists() or f.stat().st_mtime <= made for f in files)


def feed_entries(args: argparse.Namespace, scene: Scene) -> dict[str, dict]:
    """What each source that plays is fed from, as feeds.json keeps it: its file, the
    channel of it, and the files of the stems it plays (None for none)."""
    stems = stems_of(args, scene)
    out = {}
    for name, played in signals_of(args, scene).items():
        if played is None:
            continue
        mix = stems_played(played)
        out[name] = {
            "signal": str(played[0]),
            "channel": played[1],
            "stems": {k: str(stems[k]) for k in mix} if mix else None,
        }
    return out


def read_signal(
    path: Path, rate: int, read_in: dict[Path, tuple[np.ndarray, dict[bool, float]]]
) -> tuple[np.ndarray, dict[bool, float]]:
    """``path`` decoded at ``rate``, once for all the feeds (``read_in`` keeps it, with
    the gains its feeds are brought to the same RMS by)."""
    if path not in read_in:
        x, r = decoded(path)
        read_in[path] = (resampled(x, r, rate), {})
    return read_in[path]


def stems_heard(
    args: argparse.Namespace,
    scene: Scene,
    rate: int,
    read_in: dict[Path, tuple[np.ndarray, dict[bool, float]]],
) -> dict[str, np.ndarray]:
    """The stems of the recording the run plays (:func:`stems_of`), each mixed down to
    mono at ``rate``, lined up with the recording and as long as it: a free song's
    downloaded so, those of a recording of yours lined up here
    (:func:`acoustic_spaces.recordings.lined_up`)."""
    free = free_of(scene)
    files = stems_of(args, scene)
    missing = [name for name, path in files.items() if not path.exists()]
    if missing and free is not None:
        raise SystemExit(
            f"error: the stems of {free.title} are not downloaded yet; run first: "
            f"{DEMO.command('recording', args)}"
        )
    if missing:
        raise SystemExit(f"error: --stem {missing[0]}: no {files[missing[0]]}")
    stems = {name: mono(read_signal(path, rate, read_in)[0]) for name, path in files.items()}
    if free is None:
        master = recording_of(scene)
        if master is None:  # the problems say so before
            raise SystemExit("error: --stem gives the stems of a recording of yours (--recording)")
        whole = mono(read_signal(master, rate, read_in)[0])
        stems, lag = recordings.lined_up(stems, whole, rate)
        print(f"the stems line up with {master.name} {lag / rate:+.3f} s later")
    return stems


def channel_of(x: np.ndarray, channel: str | dict[str, float] | None) -> np.ndarray:
    """The ``channel`` of ``x`` (samples, or samples by channels): left or right (a mono
    file's one channel for both), or all of them mixed down (mono, or None)."""
    if channel in ("left", "right"):
        return stereo(x)[:, 0 if channel == "left" else 1]
    return mono(x)


def feeds(args: argparse.Namespace) -> None:
    """One WAV per source that plays: its file (its signal, what ``--input`` gives it, or
    the scene's recording), at the scene's sample rate, the channel the source names or
    mixed down to mono. A file mixed down is brought to one gated RMS (FEED_RMS); its
    left and right channels share one gain, which brings their mean square to FEED_RMS,
    so that its stereo balance stays and each channel's source plays at its sound power
    on average over the two, as ``auralize`` sets the balance between the sources from
    their sound powers (one gain from the mixdown played them up to 3 dB over, as two
    channels unlike each other cancel in it); sources without one are listed as
    silent."""
    scene = scene_of(args)
    rate = scene.simulation.sample_rate
    out = feeds_folder(args)
    table: dict[str, dict] = {}
    silent: list[str] = []
    length = 0
    free = free_of(scene)
    read_in: dict[Path, tuple[np.ndarray, dict[bool, float]]] = {}
    entries = feed_entries(args, scene)
    stems: dict[str, np.ndarray] | None = None
    for name, played in signals_of(args, scene).items():
        if played is None:
            silent.append(name)
            continue
        signal, channel = played
        if not signal.exists():
            if free is not None and signal == recordings.file_of(free):
                raise SystemExit(
                    f"error: {name} plays {free.title}, not downloaded yet; run first: "
                    f"{DEMO.command('recording', args)}"
                )
            raise SystemExit(f"error: {name} plays {signal}, which is not there")
        mix = stems_played(played)
        if mix:
            if stems is None:
                stems = stems_heard(args, scene, rate, read_in)
            y = sum(stems[k] * 10.0 ** (gain / 20.0) for k, gain in mix.items())
            rms = gated_rms(y, rate)
            if rms <= 0.0:
                raise SystemExit(f"error: {name}'s {channel_words(channel)} is silent")
            y = y * (FEED_RMS / rms)
        else:
            x, gains = read_signal(signal, rate, read_in)
            stereo_pair = channel in ("left", "right")
            if stereo_pair not in gains:
                rms = channels_rms(stereo(x), rate) if stereo_pair else gated_rms(x, rate)
                if rms <= 0.0:
                    raise SystemExit(f"error: {signal} is silent")
                gains[stereo_pair] = FEED_RMS / rms
            y = channel_of(x, channel) * gains[stereo_pair]
        write(out / f"{name}.wav", y, rate)
        length = max(length, len(y))
        table[name] = entries[name] | {"length_s": len(y) / rate}
        which = "" if channel is None else f", {channel_words(channel)}"
        print(f"{name:>20}: {signal.name}{which} ({len(y) / rate:.1f} s)")
    if not table:
        raise SystemExit(
            f"error: no source of {args.scene} plays anything; give one a WAV file: Plays on "
            "the source's page in the editor, signal: in the scene file, or --input "
            "SOURCE=WAV; or give the scene a recording and the sources that play it a channel"
        )
    for name in silent:
        print(f"{name:>20}: silent (no signal)")
    pipeline.write_json(
        out / FEEDS,
        {
            "rate": rate,
            "length_s": length / rate,
            "levels": FEED_LEVELS,
            "sources": table,
            "silent": silent,
        },
    )
    print(f"wrote {len(table)} feeds to {out}")


# --------------------------------------------------------------------------- the mixes


def listeners_of(args: argparse.Namespace, entries: list[dict]) -> list[str]:
    """The listeners of the tour: those ``--listeners`` names, in its order, or every
    listener in use that the results have, in the scene's order."""
    simulated = {e["listener"] for e in entries}
    names = listener_names(args.listeners)
    if names is not None:
        missing = [name for name in names if name not in simulated]
        if missing:
            raise SystemExit(
                f"error: no listener {missing[0]!r} in the results; they have "
                + ", ".join(sorted(simulated))
            )
        return names
    scene = scene_of(args)
    return [lis.name for lis in scene.listeners if lis.in_use and lis.name in simulated]


def render(args: argparse.Namespace, demo: Demo | None = None, indoors=None) -> None:
    """Every listener's mix (``--format``, binaural by default) with one gain for all, the
    loudest peak at -1 dBFS; ``auralize`` plays the feed of each source that plays in
    the run (``--sources``, :func:`~acoustic_spaces.demos.pipeline.only_playing`) at its
    sound power. The arrival time counts those sources, each at its sound power. A
    listener that a stop of the tour hears in still air has a mix in still air too
    (``audience.still``, :func:`still_stops`). When
    something moves, :func:`render_moving` makes the mixes instead. ``demo``, this one by
    default, is the demo that runs it, whose commands the messages give. ``indoors``
    (a :class:`~acoustic_spaces.demos.indoors.StillHouses`) hears each mix in the house
    its listener stands in front of, or outdoors, and writes what is felt there."""
    moving = moving_of(args)
    if moving and indoors is not None and indoors.house is not None:
        raise SystemExit(f"error: --indoors hears a scene where nothing moves; {moving[0]} does")
    if moving:
        render_moving(args, moving, demo)
        return
    table = feeds_table(args, demo)
    entries = responses_of(args, demo)
    simulation = simulations(args)[0]
    feeds = {name: feeds_folder(args) / f"{name}.wav" for name in table["sources"]}
    powers = {s.name: s.sound_power_db for s in scene_of(args).sources if s.name in feeds}
    mixes = [
        pipeline.Mix(
            name,
            simulation,
            name,
            feeds,
            weights=powers,
            info={} if indoors is None else indoors.info(name),
        )
        for name in listeners_of(args, entries)
    ]
    if not mixes:
        raise SystemExit("error: no listener to render; the scene's listeners are all left out")
    still = still_stops(args)
    if still:
        sims = simulations(args)
        mixes += [
            pipeline.Mix(
                f"{name}.{STILL}",
                sims[1],
                name,
                feeds,
                weights=powers,
                info={"atmosphere": STILL} | ({} if indoors is None else indoors.info(name)),
            )
            for name in still
            if name in {mix.listener for mix in mixes}
        ]
    post = None if indoors is None else indoors.post
    folder = run_folder(args) / "seats"
    seats = pipeline.render_mixes(mixes, folder, args, post=post)
    if indoors is not None and indoors.plant:
        from acoustic_spaces.demos.indoors import write_shakers

        rate = int(seats["rate"])
        felt = {
            seat["listener"]: indoors.shaker(seat["listener"], rate, round(seat["length_s"] * rate))
            for seat in seats["seats"].values()
        }
        seats["shaker"] = write_shakers(
            folder, {mix_stem(args, name): x for name, x in felt.items()}, rate
        )
        pipeline.write_json(folder / pipeline.SEATS, seats)


def format_of(args: argparse.Namespace, scene: Scene):
    """The format the moving mixes take: the scene's output of ``--format`` (as its
    simulation renders it; by its format, ``ambisonics``, or its label, ``ambix3``), or
    the one ``--format`` names (:func:`~acoustic_spaces.demos.pipeline.output_named`),
    through ``--hrtf`` when given. ValueError for a format it does not know."""
    from acoustic_spaces.spatial import from_output

    output = pipeline.output_named(args.format, scene.outputs)
    if args.hrtf and output.format == "binaural":
        output = output.model_copy(update={"hrtf": Path(args.hrtf)})
    return from_output(output)


def moving_done(
    args: argparse.Namespace, folder: Path, moving: list[str], names: list[str]
) -> bool:
    """Whether the moving mixes in ``folder`` are there for this run: the same trajectories,
    listeners and format, newer than the scene and the feeds."""
    path = folder / pipeline.SEATS
    if getattr(args, "again", False) or not path.exists():
        return False
    seats = pipeline.read_json(path)
    same = (
        seats.get("moving") == moving
        and seats.get("label") == pipeline.output_label(args)
        and sorted(seats.get("seats", {})) == sorted(names)
        and seats.get("trajectories") == trajectories_of(args)
    )
    made = [Path(args.scene), feeds_folder(args) / FEEDS]
    return same and all(path.stat().st_mtime >= p.stat().st_mtime for p in made if p.exists())


def trajectories_of(args: argparse.Namespace) -> dict:
    """What the run changes of the trajectories, as seats.json keeps it: the speed (km/h)
    and the step (m) it gives them, None where it keeps theirs."""
    speed, step = getattr(args, "speed", None), getattr(args, "step_m", None)
    return {
        "speed_kmh": None if speed is None else float(speed),
        "step_m": None if step is None else float(step),
    }


def levels_along(movement, source, listener: str) -> list[float | None]:
    """The A-weighted level (dB) of ``source`` (a scene source) at ``listener`` at each
    moment of ``movement``, as simulated, as a still listener's and a pass-by's are
    (:meth:`~acoustic_spaces.engine.response.Response.levels_or_held`); None at a
    moment where none of its sound arrives."""
    from acoustic_spaces.parameters import a_weighted

    power = source.power_db("octave")
    levels: list[float | None] = []
    for k in range(len(movement.times)):
        response = movement.result[movement.key(source.name, listener, k)]
        level = float(a_weighted(response.levels_or_held(power)))
        levels.append(round(level, 2) if math.isfinite(level) else None)
    return levels


def speech_along(movement, scene: Scene, source, listener: str) -> list[float | None]:
    """The STI of the speech of ``source`` (a talker of ``scene``) at ``listener`` at each
    moment of ``movement``, as the report gives a still listener's (its sound power and
    the scene's background noise: :meth:`~acoustic_spaces.scene.Scene.level_inputs`);
    None at a moment where it is not defined."""
    power, noise, talker = scene.level_inputs(source.name)
    out: list[float | None] = []
    for k in range(len(movement.times)):
        response = movement.result[movement.key(source.name, listener, k)]
        sti = float(response.speech(power, noise, talker=talker).sti)
        out.append(round(sti, 3) if math.isfinite(sti) else None)
    return out


def played_until(x: np.ndarray, rate: float, seconds: float) -> np.ndarray:
    """``x`` (samples, channels) at ``rate`` played again from its start, AGAIN_S after it
    ends, as many times as it takes to last ``seconds``, and cut there; as it is when it
    lasts that long already."""
    n = int(math.ceil(seconds * rate))
    if len(x) >= n or not len(x):
        return x
    gap = np.zeros((int(round(AGAIN_S * rate)), *x.shape[1:]), dtype=x.dtype)
    once = np.concatenate([x, gap])
    return np.concatenate([once] * math.ceil(n / len(once)))[:n]


def render_moving(args: argparse.Namespace, moving: list[str], demo: Demo | None = None) -> None:
    """Every listener's mix with ``moving`` moving, all on one clock: the scene
    simulated with every mover at each moment of its trajectory
    (:func:`~acoustic_spaces.walkthrough.simulate_movement`), the feed of each source
    that plays in the run (``--sources``) played through it at its sound power, one
    gain for all (the loudest peak at -1 dBFS), into ``moving/`` with seats.json; a
    trajectory with Doppler, which moves alone, with each early path through its
    changing delay; a recording shorter than what moves played again from its start
    (:func:`played_until`). That also keeps the moments and, at each, the A-weighted level of
    each source that plays at every listener and the STI of each talker's speech there
    (:func:`speech_along`), for the pictures; the arrival time (``arrival_s``) is left
    out where something moves, as it changes along the way, and the tour shifts no mix
    by it (``lag_s`` 0), so that the stops keep one clock."""
    from acoustic_spaces.room import Room
    from acoustic_spaces.walkthrough import (
        head_turners,
        knots,
        movers,
        render_movement,
        simulate_movement,
    )

    table = feeds_table(args, demo)
    names = listeners_of(args, responses_of(args, demo))
    if not names:
        raise SystemExit("error: no listener to render; the scene's listeners are all left out")
    folder = run_folder(args) / MOVING
    if moving_done(args, folder, moving, names):
        print(f"{folder}: there already (--again mixes them again)")
        return
    scene = moving_scene(args)
    room = Room.from_scene(scene)
    rate = int(scene.simulation.sample_rate)
    feeds = {
        name: read(feeds_folder(args) / f"{name}.wav")
        for name in playing(args, list(table["sources"]))
    }
    sounding = [s for s in scene.sources if s.name in feeds]
    fmt = format_of(args, scene)
    print(f"simulating the movement of {', '.join(moving)}, on one clock")
    movement = simulate_movement(room, progress=print, head_tracked=head_turners(scene, fmt))
    # Every listener may be heard until what moves is over and FADE_OUT_S more
    # (heard_for): a recording shorter than that plays again from its start.
    over = float(movement.times[-1])
    for name, (x, r) in list(feeds.items()):
        if len(x) and len(x) < (over + FADE_OUT_S) * r:
            feeds[name] = (played_until(x, r, over + FADE_OUT_S), r)
            print(
                f"{name} plays {len(x) / r:.1f} s, and what moves goes on for {over:.1f} s: "
                f"played again from its start, {AGAIN_S:g} s after it ends"
            )
    with_doppler = ", each early path through its changing delay" if movement.doppler else ""
    print(f"mixing {len(names)} listeners through {len(movement.times)} moments{with_doppler}")
    mixes = render_movement(movement, feeds, fmt, sample_rate=rate)
    times, places = movement.times, movement.places
    levels = {name: {s.name: levels_along(movement, s, name) for s in sounding} for name in names}
    talkers = [s for s in sounding if s.spectrum != "flat"]
    speech = {
        name: {s.name: speech_along(movement, scene, s, name) for s in talkers} for name in names
    }
    peak = max(float(abs(mixes[name]).max(initial=0.0)) for name in names)
    gain = 10.0 ** (-1.0 / 20.0) / peak if peak > 0 else 1.0
    weights = {s.name: s.sound_power_db for s in sounding}
    sources_move = any(s.name in moving for s in sounding)
    own_ends = {
        item.name: knots(item).duration_s for kind, item in movers(scene) if kind == "listeners"
    }
    entries = responses_of(args, demo)
    info = {}
    for name in names:
        x = mixes[name] * gain
        file = pipeline.mix_file(args, name)
        write(folder / file, x, rate)
        moves = sources_move or name in moving
        # Where nothing moves, the sound arrives as in the still simulation. The tour
        # shifts none of the mixes, as they keep the trajectories' one clock.
        arrival = None if moves else pipeline.arrival_s(results_of(args), entries, name, weights)
        # What moves is over for it when every source has arrived, or it has itself.
        until = float(times[-1]) if sources_move else own_ends.get(name)
        info[name] = {
            "listener": name,
            "file": file,
            "lag_s": 0.0,
            "arrival_s": arrival,
            "rms_db": level_db(x, rate, channel_weights(fmt)),
            "loudness_db": loudness_db(x, rate, channel_weights(fmt)),
            "length_s": len(x) / rate,
            "moves": moves,
            "until_s": until if moves else None,
        }
    top = max(v["rms_db"] for v in info.values())
    for name, v in info.items():
        v["relative_db"] = v["rms_db"] - top
        print(f"{name}: {v['relative_db']:+6.1f} dB(A)" + (", moving" if v["moves"] else ""))
    seats = {
        "rate": rate,
        "format": args.format,
        "label": pipeline.output_label(args),
        "gain": gain,
        "moving": moving,
        "trajectories": trajectories_of(args),
        "seats": info,
        "movement": {"times": [float(t) for t in times], "levels_a_db": levels, "sti": speech},
    }
    pipeline.write_json(folder / pipeline.SEATS, seats)
    print(f"wrote {folder}: {len(names)} mixes, {places} places along the trajectories")


# --------------------------------------------------------------------------- the video


def length_of_stop(text: str) -> dict:
    """A stop's length as ``--tour`` and ``--last-stop`` give it: phrases (a number),
    seconds (a number and s, 12s) or to the recording's end (end), as the fields of a
    :class:`~acoustic_spaces.scene.TourStop`, the others cleared. ValueError for one
    that does not read, or is not positive."""
    text = text.strip()
    if text == "end":
        return {"phrases": None, "seconds": None, "to_end": True}
    try:
        value = float(text[:-1] if text.endswith("s") else text)
    except ValueError:
        raise ValueError(f"not a length: {text!r}") from None
    if not value > 0.0:
        raise ValueError(f"a length is positive, not {text!r}")
    if text.endswith("s"):
        return {"phrases": None, "seconds": value, "to_end": False}
    return {"phrases": value, "seconds": None, "to_end": False}


def tour_text(text: str) -> list[TourStop]:
    """``--tour``'s stops: LISTENER[:still][:LENGTH],... in turn, each in the scene's
    atmosphere or in still air, for LENGTH phrases of 8 bars (a number), seconds (a
    number and s, 12s) or to the recording's end (end); a stop without one lasts as
    long as a stop does by default (:func:`timed_tour`). ValueError for one that does
    not read."""
    stops = []
    for part in text.split(","):
        fields = [field.strip() for field in part.split(":")]
        name, rest = fields[0], fields[1:]
        given: dict = {"listener": name}
        if rest and rest[0] in ("", *ATMOSPHERES):
            given["atmosphere"] = rest.pop(0) or "scene"
        if rest and rest[0]:
            length = rest.pop(0)
            try:
                given |= length_of_stop(length)
            except ValueError:
                rest.append(length)
        if not name or rest:
            raise ValueError(
                f"--tour takes LISTENER[:still][:LENGTH],... (LENGTH in phrases, as 2, in "
                f"seconds, as 12s, or end), not {part!r}"
            )
        try:
            stops.append(TourStop(**given))
        except ValueError as exc:  # a length of 0, say
            raise ValueError(f"--tour: {part!r}: {exc.errors()[0]['msg']}") from None
    return stops


def tour_given(args: argparse.Namespace, scene: Scene) -> list[TourStop] | None:
    """The tour of a run where nothing moves or the timeline is shared: ``--tour``'s
    (:func:`tour_text`), in its own order, without the stops at listeners
    ``--listeners`` leaves out, as in every demo; else the scene's (``tour``), its stops
    at the listeners ``--listeners`` names in that order (a listener the tour does not
    visit, in a stop of its own); None when neither gives one. ``scene`` is the scene as
    the run plays it: in its still air, a stop in still air is in its only air."""
    given = getattr(args, "tour", None)
    stops = tour_text(given) if given else scene.tour
    if stops is None:
        return None
    names = listener_names(getattr(args, "listeners", None))
    if names is not None and given:
        stops = [stop for stop in stops if stop.listener in names]
    elif names is not None:
        # A tour that ends back where it began ends at the first listener named.
        end = stops[-1]
        closing = len(stops) > 1 and end.listener == stops[0].listener
        body = stops[:-1] if closing else stops
        stops = [
            stop
            for name in names
            for stop in [s for s in body if s.listener == name] or [TourStop(listener=name)]
        ]
        if closing:
            stops.append(end.model_copy(update={"listener": names[0]}))
    last = getattr(args, "last_stop", None)
    if last and stops:
        stops[-1] = stops[-1].model_copy(update=length_of_stop(last))
    if scene.still:
        stops = [stop.model_copy(update={"atmosphere": "scene"}) for stop in stops]
    return stops


def stop_key(stop: TourStop) -> str:
    """The mix a stop of the tour plays: its listener's, or in still air its listener's
    in still air (``audience.still``)."""
    return stop.listener if stop.atmosphere == "scene" else f"{stop.listener}.{STILL}"


def tempo_of(args: argparse.Namespace, scene: Scene) -> tuple[float, float] | None:
    """The tempo the tour's phrases go by (beats a minute) and when the first downbeat
    falls (s): ``--bpm`` and ``--downbeat`` for a recording of yours, else the free
    recording's that a source of the run plays; None when neither has one."""
    bpm = getattr(args, "bpm", None)
    if bpm is not None:
        return float(bpm), float(getattr(args, "downbeat", None) or 0.0)
    free = free_of(scene)
    if free is None or free.bpm is None:
        return None
    file = recordings.file_of(free)
    played = [v[0] for v in signals_of(args, scene).values() if v is not None]
    return (float(free.bpm), float(free.downbeat_s)) if file in played else None


def timed_tour(
    stops: list[TourStop],
    length_s: float,
    tempo: tuple[float, float] | None,
    seconds: float | None = None,
) -> list[tuple[str, float, float]]:
    """(mix, start, end) of each stop in the recording's time, one after another from
    its start: a stop in phrases ends that many phrases of 8 bars (``tempo``: beats a
    minute and the first downbeat, s) after the phrase it starts in, so that the cuts
    fall where the music turns; one in seconds after as many; one to the end at the
    recording's end (``length_s``); one that gives no length after ``seconds``, else
    SECONDS, and where no stop gives one and the recording is too short for that, each
    as long, sharing its length out. A stop with no time left is left out, the last one
    is cut at the end, and stops of one mix in a row are one. ValueError for a stop in
    phrases without a tempo."""
    from acoustic_spaces.demos import music

    each = SECONDS if seconds is None else seconds
    if stops and all(s.phrases is None and s.seconds is None and not s.to_end for s in stops):
        shared = length_s / len(stops)
        if shared < each - 1e-6:
            print(f"the piece lasts {length_s:.0f} s: each listener is heard {shared:.1f} s")
            each = shared
    out: list[tuple[str, float, float]] = []
    t = 0.0
    for stop in stops:
        if t >= length_s - 1e-9:
            break
        if stop.to_end:
            end = length_s
        elif stop.phrases is not None:
            if tempo is None:
                raise ValueError(
                    f"the tour's stop at {stop.listener} lasts {stop.phrases:g} phrases, and "
                    "the recording has no tempo: give it seconds (12s), or the tempo (--bpm)"
                )
            phrase = music.phrase_s({"bpm": tempo[0]})
            first = tempo[1]
            n = max(math.floor((t - first) / phrase + 1e-6), 0)
            end = first + (n + stop.phrases) * phrase
            if end <= t + 1e-6:
                end = first + (n + 1) * phrase
        else:
            end = t + (stop.seconds or each)
        end = min(end, length_s)
        key = stop_key(stop)
        if out and out[-1][0] == key:
            out[-1] = (key, out[-1][1], end)
        elif end > t:
            out.append((key, t, end))
        t = end
    return out


def still_stops(args: argparse.Namespace) -> list[str]:
    """The listeners a stop of the run's tour hears in still air, beside the scene's own
    atmosphere, in the order they are first heard so: none where something moves and
    each listener is heard from the clock's start, or where the run hears the scene in
    still air already."""
    scene = _run(args)
    if scene is None or scene.still:
        return []
    moving = moving_of(args)
    if moving and timeline_of(args, moving) != "shared":
        return []
    try:
        stops = tour_given(args, scene) or []
    except ValueError:
        return []
    return list(dict.fromkeys(s.listener for s in stops if s.atmosphere == STILL))


def still_args(args: argparse.Namespace) -> argparse.Namespace:
    """The run's arguments in still air (``--atmosphere still``)."""
    return argparse.Namespace(**{**vars(args), "atmosphere": STILL})


def length_of(table: dict, names: list[str]) -> float:
    """How long the feeds of the sources ``names`` play (s), by ``table`` (the feeds'
    FEEDS): the longest of them; that of every feed when ``names`` is empty."""
    return max(
        (float(table["sources"][name]["length_s"]) for name in names),
        default=float(table["length_s"]),
    )


def tour_of(
    args: argparse.Namespace, seats: dict, length_s: float, demo: Demo | None = None
) -> list[pipeline.TourStop]:
    """The stops, on the run's timeline (:func:`timeline_of`): on a shared one, the
    run's tour (:func:`tour_given`, :func:`timed_tour`) where it has one, else
    :func:`shared_stops`; else :func:`stops_from_start`; for ``--seconds`` or as long
    as the timeline has them."""
    entries = responses_of(args, demo)
    names = [name for name in listeners_of(args, entries) if name in seats["seats"]]
    if not names:
        run = (demo or DEMO).command("render", args)
        raise SystemExit(f"error: no mix to tour; run first: {run}")
    given = getattr(args, "seconds", None)
    seconds = None if given is None else float(given)
    if timeline_of(args, seats.get("moving") or []) == "shared":
        scene = scene_of(args)
        tour = tour_given(args, scene)
        if tour is not None:
            timed = timed_tour(tour, length_s, tempo_of(args, scene), seconds)
            keys = list(dict.fromkeys(key for key, _, _ in timed))
            pipeline.missing_mixes(seats, keys, (demo or DEMO).command("render", args))
            gains = stops_gains(seats, keys)
            return [pipeline.TourStop(key, a, a, b, gains[key]) for key, a, b in timed]
        return shared_stops(names, pipeline.gains_of(seats, names), seconds, length_s)
    return stops_from_start(names, seats, pipeline.gains_of(seats, names), seconds)


def stops_gains(seats: dict, keys: list[str]) -> dict[str, float]:
    """The gain (dB) each mix of the tour ``keys`` plays at (:func:`pipeline.gains_of`),
    a listener's in still air at the gain of the same listener in the scene's own
    atmosphere, so that the difference the weather makes is heard as simulated."""
    twins = {key: key.removesuffix(f".{STILL}") for key in keys}
    twins = {key: twin if twin in seats["seats"] else key for key, twin in twins.items()}
    gains = pipeline.gains_of(seats, list(dict.fromkeys([*keys, *twins.values()])))
    return {key: gains[twins[key]] for key in keys}


def shared_stops(
    names: list[str], gains: dict[str, float], seconds: float | None, length_s: float
) -> list[pipeline.TourStop]:
    """The stops of a shared timeline without a tour: each listener of ``names`` in
    turn for ``seconds`` (SECONDS when None) while the piece runs on, crossfaded into
    the next; a piece too short for that (``length_s``) shares its length out
    (:func:`timed_tour`)."""
    timed = timed_tour([TourStop(listener=name) for name in names], length_s, None, seconds)
    return [pipeline.TourStop(key, a, a, b, gains[key]) for key, a, b in timed]


def stops_from_start(
    names: list[str], seats: dict, gains: dict[str, float], seconds: float | None
) -> list[pipeline.TourStop]:
    """The stops of a timeline from each listener's start: each listener of ``names``
    from the clock's start, for ``seconds`` or, None, until what moves is over for it
    and FADE_OUT_S more (:func:`heard_for`); each as long as its mix at most, faded out
    over its last FADE_OUT_S (half of it at most) and GAP_S before the next."""
    stops, at = [], 0.0
    for name in names:
        heard = heard_for(seats, name) if seconds is None else seconds
        length = min(heard, float(seats["seats"][name]["length_s"]))
        stops.append(
            pipeline.TourStop(
                name,
                at,
                0.0,
                length,
                gains[name],
                joined_before=False,
                joined_after=False,
                fade_s=min(FADE_OUT_S, length / 2),
            )
        )
        at += length + GAP_S
    return stops


def heard_for(seats: dict, name: str) -> float:
    """How long the listener ``name`` is heard from the clock's start where ``--seconds``
    says nothing (s): until what moves is over for it (``seats``' ``until_s`` of it:
    every moving source arrived, or it has itself where only listeners move) and
    FADE_OUT_S more; SECONDS where nothing moves for it, as for a listener standing
    still while another walks. Mixes made before ``until_s`` was kept hear every
    listener until every mover has arrived."""
    seat = seats["seats"][name]
    movement = seats.get("movement")
    if "until_s" in seat:
        until = seat["until_s"]
    elif seats.get("moving") and movement:
        until = movement["times"][-1]
    else:
        until = None
    return SECONDS if until is None else float(until) + FADE_OUT_S


def video(args: argparse.Namespace, demo: Demo | None = None, indoors=None) -> None:
    """The run's audio (``<run>.wav``), one still per stop, the thumbnail, and the video
    (``<run>.mp4``): the stills with a scrolling spectrogram in the panel at their foot
    (ffmpeg, unless ``--no-mp4``). When
    something moves, the video is a moving picture instead: the movers go along their
    trajectories as the piece plays, with a still of each stop. ``demo``, this one by default,
    is the demo that runs it: its commands and its settings go in the messages and on
    the stills. ``indoors`` (a :class:`~acoustic_spaces.demos.indoors.StillHouses`),
    the mixes heard in a house at each listener: the stills give the levels there."""
    demo = demo or DEMO
    from acoustic_spaces.demos import pictures, scene_pictures
    from acoustic_spaces.room import Room, outline_edges
    from acoustic_spaces.walkthrough import knots, movers

    table = feeds_table(args, demo)
    moving = moving_of(args)
    folder = run_folder(args) / (MOVING if moving else "seats")
    seats = pipeline.seats_of(folder, demo.command("render", args), args)
    if seats.get("moving", []) != moving:
        raise SystemExit(
            f"error: the mixes in {folder} are of other trajectories; run first: "
            f"{demo.command('render', args)}"
        )
    # The mixes heard where the stills say: outdoors, or in the houses --indoors asks.
    if any(
        seat.get("post") != (None if indoors is None else indoors.info(key).get("post"))
        for key, seat in seats["seats"].items()
    ):
        raise SystemExit(
            f"error: the mixes in {folder} were made for another --indoors; "
            f"run first: {demo.command('render', args)}"
        )
    fed = playing(args, list(table["sources"]))  # those that play in the run (--sources)
    stops = tour_of(args, seats, length_of(table, fed), demo)
    audio = pipeline.tour_audio(
        stops,
        folder,
        seats,
        crossfade_s=CROSSFADE_S,
        fade_out_s=min(FADE_OUT_S, stops[-1].length / 2),
        scene=args.scene,
    )
    sound = product(args, ".wav")
    write(sound, audio, seats["rate"])
    print(f"wrote {sound} ({len(audio) / seats['rate']:.1f} s)")
    if seats.get("shaker"):
        from acoustic_spaces.demos.indoors import tour_shaker

        tour_shaker(
            product(args, "_shaker.wav"),
            stops,
            folder,
            seats,
            seats["shaker"],
            CROSSFADE_S,
            min(FADE_OUT_S, stops[-1].length / 2),
        )
    scene = moving_scene(args)
    data = scene.to_data()
    report = pipeline.read_json(results_of(args) / pipeline.PARAMETERS)
    variants = {}
    if still_stops(args):
        still = results_at(args, run_scene_file(still_args(args))) / pipeline.PARAMETERS
        variants = {
            f"{name}.{STILL}": (name, pipeline.read_json(still)) for name in still_stops(args)
        }
    room = Room.from_scene(scene)
    outline = outline_edges(room)[0]
    drawing = scene_pictures.Pictures(
        data,
        outline,
        report if indoors is None else indoors.report(report),
        seats["seats"],
        stops,
        playing=fed,
        credit=args.credit or plays_line(table, free_of(scene), fed),
        tool=pictures.settings_lines(demo, args, data, report),
        fmt=args.format,
        paths={(kind, item.name): knots(item) for kind, item in movers(scene)},
        movement=seats.get("movement"),
        indoors=None if indoors is None else indoors.pictures(),
        variants=variants,
        room=room,
        hide=hidden(args),
    )
    pictures.save(drawing.thumbnail(), product(args, "_thumbnail.png"))
    if moving:
        pipeline.animation(
            run_folder(args),
            len(audio) / seats["rate"],
            drawing.frame,
            sound,
            out=product(args, ".mp4"),
            stills=[(stop.key, stop.at + min(0.5, stop.length / 2)) for stop in stops],
            no_mp4=args.no_mp4,
            shown=drawing.shown_at,
        )
        return
    # Each still lasts until the next stop: through the silence after it, between stops
    # heard from each listener's start.
    ends = [stop.at for stop in stops[1:]] + [stops[-1].end]
    pipeline.video(
        run_folder(args),
        [(stop.key, end - stop.at) for stop, end in zip(stops, ends, strict=True)],
        drawing.still,
        sound,
        out=product(args, ".mp4"),
        no_mp4=args.no_mp4,
        shown=drawing.shown,
    )


def plays_line(
    table: dict,
    free: recordings.Recording | None = None,
    names: list[str] | None = None,
) -> str:
    """What the sources ``names`` play (None: every one fed), for the stills when
    ``--credit`` says nothing: the credit the free recording's licence asks for, when
    one of them plays it, and each other file by source."""
    kept = str(recordings.file_of(free)) if free is not None else None
    fed = {name: v for name, v in table["sources"].items() if names is None or name in names}
    parts = (
        [free.credit] if free is not None and any(v["signal"] == kept for v in fed.values()) else []
    )
    others = [f"{name}: {Path(v['signal']).name}" for name, v in fed.items() if v["signal"] != kept]
    if others:
        parts.append("Playing " + ", ".join(others))
    return " · ".join(parts)


DEMO = Demo(
    name="scene",
    example="",
    title=N_("This scene"),
    description=__doc__,
    script=Path(__file__),
    steps=[
        Step(
            "recording",
            recording,
            "download the scene's free recording, if it names one and it is not there",
            lambda args: recording_needed(args) is None,
            recording_download,
        ),
        Step(
            "feeds",
            feeds,
            "one WAV per source that plays: its file, its channel or mono, at one gated RMS",
            feeds_done,
        ),
        SIMULATE,
        Step("render", render, "every listener, with one gain for all"),
        Step("video", video, "the run's video, its audio and the stills, named after the run"),
    ],
    options=[
        Option(
            "scene",
            "the scene to demo: its sources play their signals",
            None,
            "file",
            filter="Scene files (*.yaml *.yml)",
            named="never",  # its name begins the run's (stem_of)
        ),
        Option(
            "recording",
            "what the sources with a channel play in this run, in place of the scene's "
            "own recording (Recording on the scene's page): a free one, a song or a talker "
            "of the ACE corpus, downloaded the first time, or a file of yours (WAV, or "
            "anything ffmpeg decodes); a source with a speech spectrum that plays a talker "
            "takes its talker's, a woman's or a man's (default: the scene's recording)",
            None,
            "choice",
            choices=recordings.NAMES,
            or_file=True,
            steps=("recording", "feeds", "simulate", "render", "video"),
            unset="the scene's",
            named="plays",
            moot=lambda args: (
                None
                if recording_heard(args)
                else "no source plays the scene's recording: give one a channel"
            ),
        ),
        Option(
            "seconds",
            "how long each listener is heard (s): on a shared timeline 20 s by default, and "
            "a piece too short for every listener shares its length out; from each "
            "listener's start, by default until what moves is over for it (every moving "
            "source arrived, or the listener itself) and 3 s more, 20 s where nothing "
            "moves for it",
            None,
            "float",
            steps=("video",),
            unit="s",
            unset="as the timeline has it",
        ),
        Option(
            "timeline",
            "shared: one clock, the tour going from listener to listener as the piece and "
            "the movers go on; per-listener: each listener heard from the clock's start, "
            "the whole of what moves; auto: per-listener when something moves, shared "
            "otherwise",
            "auto",
            "choice",
            choices=TIMELINES,
            steps=("video",),
        ),
        dataclasses.replace(
            listeners_option(),
            help="only these listeners, in this order, comma separated (default: every "
            "listener in use, in the scene's order): the demo mixes those alone and its "
            "video visits them in turn, with their stops of the tour where there is one; "
            "the simulation keeps every listener, so another choice needs no new one",
        ),
        Option(
            "tour",
            "the stops the video plays in turn where the timeline is shared, in place of "
            "the scene's own tour, as LISTENER[:still][:LENGTH],...: a listener, in still "
            "air with still, for LENGTH phrases of 8 bars of 4 beats (2), seconds (12s) or "
            "to the recording's end (end); a stop without one lasts --seconds (default: "
            "the scene's tour, else every listener in turn)",
            None,
            steps=("render", "video"),
            unset="the scene's",
        ),
        Option(
            "last_stop",
            "how long the tour's last stop lasts, in place of what the tour says: phrases "
            "of 8 bars of 4 beats (2), seconds (12s) or to the recording's end (end)",
            None,
            steps=("video",),
            unset="as the tour says",
        ),
        Option(
            "atmosphere",
            "the atmosphere the run hears the scene in: scene, its own (wind, temperature "
            "profiles, weather zones), or still, still air, where sound goes straight; the "
            "scene so changed is simulated once (default: the scene's own, and still air "
            "where a stop of the tour says so)",
            None,
            "choice",
            choices=ATMOSPHERES,
            steps=("simulate", "render", "video"),
            unset="the scene's own",
            moot=lambda args: (
                "the scene's air is still"
                if (found := _current(getattr(args, "scene", None))) is not None and found.still
                else None
            ),
        ),
        *sources_options(help=SOURCES_HELP, steps=("render", "video")),
        Option(
            "vocal_effort",
            "how loud every talker in use (a source with a speech spectrum) talks in this "
            "run, as ISO 9921 has it: relaxed, normal, raised or loud, its speech 54, 60, "
            "66 or 72 dB(A) 1 m in front of it; the scene so changed is simulated once "
            "(default: as in the scene)",
            None,
            "choice",
            choices=VOCAL_EFFORTS,
            steps=("simulate", "render", "video"),
            unset="as in the scene",
            aliases=("--voice",),
            moot=lambda args: None if talkers_of(args) else "no source in use talks",
        ),
        Option(
            "credit",
            "a line on the stills about the recordings: who made them, under which licence",
            None,
            steps=("video",),
            named="never",
        ),
        Option(
            "hide",
            "what the stills of an outdoor scene leave out, comma-separated: side, the view "
            "from the side through the source heard loudest and you, with the sound's rays; "
            "wind, the wind and the weather zones from above; ground, the ground's materials "
            "and what stands on it, from above (default: none, they show it all)",
            None,
            steps=("video",),
            unset="nothing",
            moot=lambda args: (
                "the scene is a room"
                if (found := _current(getattr(args, "scene", None))) is not None
                and found.geometry.ground is None
                else None
            ),
        ),
        Option(
            "bpm",
            "the tempo of a recording of yours (beats a minute), which the tour's stops in "
            "phrases of 8 bars go by; a free song has its own",
            None,
            "float",
            steps=("video",),
            unit="bpm",
            unset="the recording's",
        ),
        Option(
            "downbeat",
            "when the first downbeat of a recording of yours falls (s), which its phrases "
            "run from (with --bpm)",
            None,
            "float",
            steps=("video",),
            unit="s",
            unset="at its start",
        ),
        Option(
            "stems",
            "a stem of a recording of yours (--recording FILE), as STEM=FILE, once per stem "
            f"({', '.join(recordings.STEM_NAMES)}; WAV, or anything ffmpeg decodes): what a "
            "source whose channel names it, or a mix of stems, plays, lined up with the "
            "recording; a stem not given is left out of the mixes, and a source none of "
            "whose stems is given is silent (a free song has its own)",
            None,
            "files",
            recordings.STEM_NAMES,
            ("feeds",),
            flag="--stem",
            filter=AUDIO_FILTER,
            moot=lambda args: stems_moot(args),
        ),
        Option(
            "moving",
            "the sources and listeners that move along their trajectories in this run, by name, "
            "comma separated, every other mover standing where it starts; none keeps them "
            "all still (default: those whose trajectory is on in the scene, but a source "
            "--sources leaves out, which stands)",
            None,
            "movers",
            steps=("render", "video"),
        ),
        Option(
            "stand",
            "where a listener stands in this run, as LISTENER=X,Y (m), its height kept, "
            "facing the nearest source in use, its trajectory moved and turned with it; "
            "the scene so changed is simulated once (default: where the scene has it)",
            None,
            steps=("simulate", "render", "video"),
        ),
        Option(
            "turn",
            "a listener turning on the spot in this run, in place of its trajectory, as "
            "LISTENER=T:YAW,T:YAW,...: the yaw (degrees, turning left from the way it "
            "faces) it has at each time (s), turning evenly in between; binaurally heard "
            "through the head tracker (default: as the scene has it)",
            None,
            steps=("render", "video"),
        ),
        Option(
            "speed",
            "the speed every trajectory that moves goes at in this run (km/h), along its "
            "points, or through its keyframes' places in turn (default: as in the scene)",
            None,
            "float",
            steps=("render", "video"),
            unit="km/h",
            unset="as in the scene",
            moot=lambda args: None if moving_of(args) else "nothing moves",
        ),
        Option(
            "step_m",
            "how far apart along each trajectory that moves the sound is simulated in this "
            "run (m): finer is smoother and slower (default: each trajectory's own step, "
            "0.25 m unless it gives one)",
            None,
            "float",
            flag="--step",
            steps=("render", "video"),
            unit="m",
            unset="each trajectory's own",
            moot=lambda args: None if moving_of(args) else "nothing moves",
        ),
        Option(
            "work",
            "folder for everything the demo writes (default: demo beside the scene)",
            None,
            "folder",
        ),
        dataclasses.replace(
            input_option(("feeds",)),
            help="a WAV file a source plays instead of its signal (Plays), as SOURCE=WAV, "
            "once per source, fed as its signal would be (at the source's sound power); a "
            "silent source may be given one too",
        ),
        *[
            dataclasses.replace(o, help=HRTF_HELP) if o.name == "hrtf" else o
            for o in common_options()
            if o.name != "work"
        ],
    ],
    simulations=simulations,
    problems=problems,
    program="acoustic-spaces demo",
    leading="scene",
    module="acoustic_spaces.demos.scene_demo",
    work_of=work_of,
    completes=completed,
    stem_of=stem_of,
    plays=plays,
    moving=moving_of,
    trajectories=moving_scene,
    listeners=lambda args: listeners_in(args.scene),
    sources=sources_of,
)


def preset(
    name: str,
    example: str,
    title: str,
    description: str,
    script: Path,
    *,
    defaults: dict | None = None,
    renamed: dict | None = None,
    steps: dict[str, tuple[str, ...]] | None = None,
) -> Demo:
    """An example's script as a preset of the demo of any scene: :data:`DEMO` on the
    example's own scene (``scene.yaml`` beside the script), with the defaults the script
    has (``defaults``, by option), its flags read as this demo's (``renamed``, by option:
    each old flag and what a value of it is in that option's,
    :attr:`~acoustic_spaces.demos.Option.renamed`) and its steps run as the ones that
    took them over (``steps``, by step: the old names), so that the command lines written
    for the script keep working."""
    defaults, renamed, steps = defaults or {}, renamed or {}, steps or {}
    unknown = sorted(
        ({*defaults, *renamed} - {o.name for o in DEMO.options})
        | ({*steps} - {s.name for s in DEMO.steps})
    )
    if unknown:
        raise ValueError(f"{name}: the demo of any scene has no {', '.join(unknown)}")

    def option(given: Option) -> Option:
        changes: dict = {}
        if given.name == "scene":
            changes["default"] = Path("scene.yaml")
        if given.name in defaults:
            changes["default"] = defaults[given.name]
        if given.name in renamed:
            changes["renamed"] = (*given.renamed, *renamed[given.name])
        return dataclasses.replace(given, **changes) if changes else given

    return dataclasses.replace(
        DEMO,
        name=name,
        example=example,
        title=title,
        description=description,
        script=script,
        steps=[
            dataclasses.replace(s, aliases=(*s.aliases, *steps[s.name])) if s.name in steps else s
            for s in DEMO.steps
        ],
        options=[option(o) for o in DEMO.options],
        program=None,
        leading=None,
        module=None,
    )


if __name__ == "__main__":
    sys.exit(DEMO.main())
