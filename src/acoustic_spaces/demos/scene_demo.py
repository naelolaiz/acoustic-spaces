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
Each file is brought to one gated RMS (its channels mixed down) and every channel taken
out of it keeps that gain, so a recording's stereo balance stays; then each source's
sound power in the scene (``level_db``) sets how loud it is against the others, as the
simulation says and not as the recordings happen to be. Sources with neither stay
silent. Every listener in use is heard in turn, in the scene's order (or only those
``--listeners`` names, in its order), for ``--seconds`` each while the piece runs on,
with one gain for all the stops and a crossfade between them, binaural through the HRTFs
of the scene's binaural output (or ``--hrtf``, or a spherical head model when neither
names any). The stills show the scene from above with the listener of the moment, the
level there, how long the sound takes to arrive and T30 at mid frequencies, and say how
the sound was made. The simulation keeps every listener in use, so another choice of
``--listeners`` needs no new one.

Sources and listeners whose walk is on move along it while the piece plays, all on one
clock from its start (:func:`acoustic_spaces.walkthrough.simulate_movement`): every
listener is heard as it walks or as the sources walk past it, and the video moves them
as they go, its captions giving the level at each moment. ``--moving`` picks the walks
of a run without changing the scene: ``--moving visitor,guide`` moves those two and
leaves every other walker standing where it starts, ``--moving none`` keeps them all
still, so one scene serves both. A walk with Doppler (a car going past) moves alone:
every listener hears each early path through its own delay as it changes, so the pitch
shifts as it passes, the late sound crossfaded from place to place
(:func:`acoustic_spaces.walkthrough.render_movement`), all from one simulation.

    acoustic-spaces demo scene.yaml              # every step, into demo/ beside the scene
    acoustic-spaces demo scene.yaml recording    # download the scene's free recording
    acoustic-spaces demo scene.yaml feeds        # one step
    acoustic-spaces demo scene.yaml --hrtf me.sofa --seconds 15 --listeners front,back
    acoustic-spaces demo scene.yaml --input talker=other_take.wav
    acoustic-spaces demo scene.yaml --moving none   # every walker standing where it starts

The editor runs it from **Simulation > Run demo…** as *This scene*, with a tick for each
walk in its Movement table and for each listener in its Listeners list. The video needs
ffmpeg on the PATH and Pillow (the ``viz`` extra). The five demos of the examples are
scripts of their own, with their recordings; this one takes any scene.
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
    Step,
    common_options,
    input_option,
    listener_names,
    listeners_in,
    listeners_option,
    moving_names,
    pipeline,
)
from acoustic_spaces.demos.audio import (
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
from acoustic_spaces.i18n import N_
from acoustic_spaces.scene import Scene

#: What the feeds are brought to (gated RMS, full scale): the loudness every source's
#: signal has before the simulation's sound powers set the balance.
FEED_RMS = 0.1
#: The feeds, and what each source plays, in the work folder.
FEEDS = "feeds.json"
#: --hrtf here: a scene names its own HRTFs in its binaural output.
HRTF_HELP = (
    "binaural mixes through this SOFA file of HRTFs, your own or measured, instead of the "
    "one the scene's binaural output names (or a spherical head model when it names none)"
)
#: What a source plays: the file, and the channel of it (None: mixed down to mono).
Played = tuple[Path, str | None]
#: How :func:`plays` names a channel.
CHANNEL_WORDS = {"left": "left channel", "right": "right channel", "mono": "mixed down to mono"}
CROSSFADE_S = 1.0
FADE_OUT_S = 3.0
#: The folder of the mixes when something moves (the still ones go to ``seats``).
MOVING = "moving"


def work_of(args: argparse.Namespace) -> Path:
    """Where the demo writes: ``--work``, or ``demo`` beside the scene."""
    return Path(args.work) if args.work else Path(args.scene).parent / "demo"


@functools.lru_cache(maxsize=8)
def _scene_at(path: str, _changed: int) -> Scene | None:
    """The scene at ``path`` as it is now (``_changed``, its time of change, keeps the
    cache fresh), or None when it cannot be read: :func:`problems` says why."""
    try:
        return Scene.from_yaml(path)
    except Exception:
        return None


def _current(scene: str | Path | None) -> Scene | None:
    if not scene or not Path(scene).exists():
        return None
    return _scene_at(str(Path(scene).resolve()), Path(scene).stat().st_mtime_ns)


def _scene_hrtf(scene: str | Path | None) -> Path | None:
    """The HRTFs the binaural output of the scene names, or None."""
    found = _current(scene)
    if found is None:
        return None
    return next((o.hrtf for o in found.outputs if o.format == "binaural" and o.hrtf), None)


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


def signals_of(args: argparse.Namespace, scene: Scene) -> dict[str, Played | None]:
    """What each source in use plays, and which channel of it: the file ``--input``
    gives it, else its signal, else, with a channel, the scene's recording (None:
    silent)."""
    inputs = getattr(args, "inputs", None) or {}
    recording = recording_of(scene)
    out: dict[str, Played | None] = {}
    for source in scene.sources:
        if not source.in_use:
            continue
        file = inputs.get(source.name) or source.signal
        if file is None and source.channel is not None:
            file = recording
        out[source.name] = None if file is None else (Path(file), source.channel)
    return out


def plays(args: argparse.Namespace) -> dict[str, str]:
    """Every source in use and what it plays unless ``--input`` gives it a file: the name
    of its signal or of the scene's recording, and the channel, or silent."""
    found = _current(getattr(args, "scene", None))
    if found is None:
        return {}
    free = free_of(found)
    free_file = recordings.file_of(free) if free is not None else None
    out = {}
    own = argparse.Namespace(**{**vars(args), "inputs": None})
    for name, played in signals_of(own, found).items():
        if played is None:
            out[name] = "silent"
            continue
        file, channel = played
        what = free.title if free is not None and file == free_file else file.name
        out[name] = what if channel is None else f"{what}, {CHANNEL_WORDS[channel]}"
    return out


def recording_needed(args: argparse.Namespace) -> recordings.Recording | None:
    """The free recording the run plays and has not downloaded yet, or None."""
    scene = _current(getattr(args, "scene", None))
    free = free_of(scene) if scene is not None else None
    if free is None:
        return None
    path = recordings.file_of(free)
    if path.exists():
        return None
    played = [v[0] for v in signals_of(args, scene).values() if v is not None]
    return free if path in played else None


def recording_download(args: argparse.Namespace) -> str | None:
    """What ``recording`` downloads: the free recording, its licence and size."""
    free = recording_needed(args)
    if free is None:
        return None
    return f"{free.title} ({free.licence}, {free.download_mb:g} MB), kept for every scene"


def recording(args: argparse.Namespace) -> None:
    """The free recording the scene names, downloaded into the recordings folder
    (:func:`acoustic_spaces.recordings.data_folder`) unless it is there or nothing plays
    it."""
    free = recording_needed(args)
    if free is None:
        scene = _current(args.scene)
        named = free_of(scene) if scene is not None else None
        if named is None:
            print("the scene names no free recording: nothing to download")
        else:
            print(f"{named.title}: there already, or nothing plays it")
        return
    path = recordings.get(free)
    print(f"{free.title}: {path}")


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


def moving_scene(args: argparse.Namespace) -> Scene:
    """The scene as this run plays it: with ``--moving``, those walk and every other
    walker stands where it starts (:func:`~acoustic_spaces.walkthrough.with_moving`)."""
    from acoustic_spaces.walkthrough import with_moving

    return with_moving(scene_of(args), moving_names(getattr(args, "moving", None)))


def moving_of(args: argparse.Namespace) -> list[str]:
    """Who moves in this run: the sources and listeners in use whose walks are on, by
    name, listeners first; none when the scene cannot be read or ``--moving`` names
    what does not walk (:func:`problems` says why)."""
    from acoustic_spaces.walkthrough import movers, with_moving

    scene = _current(getattr(args, "scene", None))
    if scene is None:
        return []
    try:
        scene = with_moving(scene, moving_names(getattr(args, "moving", None)))
    except ValueError:
        return []
    return [item.name for _kind, item in movers(scene)]


def results_of(args: argparse.Namespace) -> Path:
    return work_of(args) / "results"


def scene_of(args: argparse.Namespace) -> Scene:
    """The scene, its sources' signals resolved against its folder."""
    return Scene.from_yaml(args.scene)


def simulations(args: argparse.Namespace) -> list[pipeline.Simulation]:
    return [pipeline.Simulation(Path(args.scene), results_of(args), args.format)]


def responses_of(args: argparse.Namespace, demo: Demo | None = None) -> list[dict]:
    """The responses of the scene's simulation; ``demo`` (this one by default) says how
    to run it when they are not there."""
    run = (demo or DEMO).command("simulate", args)
    return pipeline.responses_of(simulations(args)[0], args, run)


def problems(demo: Demo, args: argparse.Namespace) -> str | None:
    """A scene is needed, it must be there, and a source of it must play a WAV file
    that is there (its signal, or the one ``--input`` gives it)."""
    if not getattr(args, "scene", None):
        return "give the scene to demo (--scene)"
    resolved = demo.resolved(args)
    path = Path(resolved.scene)
    if not path.exists():
        return f"no {path}"
    try:
        scene = Scene.from_yaml(path)
    except Exception as exc:  # the scene's own message
        return f"{path.name}: {exc}"
    signals = {k: v for k, v in signals_of(resolved, scene).items() if v is not None}
    if not signals:
        return (
            "no source plays anything: give one a WAV file (Plays on its page in the "
            "editor, signal: in the scene file, or --input SOURCE=WAV), or give the scene "
            "a recording and the sources that play it a channel"
        )
    free = free_of(scene)
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
        moving = with_moving(scene, moving_names(getattr(resolved, "moving", None)))
    except ValueError as exc:
        return f"--moving: {exc}"
    walkers = [item for _kind, item in movers(moving)]
    for item in walkers:
        if item.walk.doppler and len(walkers) > 1:
            return (
                f"{item.name} walks with Doppler, which plays alone: leave the others "
                f"standing (--moving {item.name}), or turn its Doppler off"
            )
    return None


# --------------------------------------------------------------------------- the feeds


def feeds_table(args: argparse.Namespace, demo: Demo | None = None) -> dict:
    path = work_of(args) / "feeds" / FEEDS
    if not path.exists():
        raise SystemExit(f"error: no feeds yet; run first: {(demo or DEMO).command('feeds', args)}")
    return pipeline.read_json(path)


def feeds_done(args: argparse.Namespace) -> bool:
    """Whether the feeds are there, play what the scene and ``--input`` ask for, and are
    newer than the scene and the files they play."""
    path = work_of(args) / "feeds" / FEEDS
    if not path.exists():
        return False
    made = path.stat().st_mtime
    table = pipeline.read_json(path)
    scene = _current(args.scene)
    if scene is None:
        return False
    wanted = {k: [str(v[0]), v[1]] for k, v in signals_of(args, scene).items() if v is not None}
    if wanted != {k: [v["signal"], v.get("channel")] for k, v in table["sources"].items()}:
        return False
    files = [Path(args.scene), *(Path(v[0]) for v in wanted.values())]
    return all(not f.exists() or f.stat().st_mtime <= made for f in files)


def channel_of(x: np.ndarray, channel: str | None) -> np.ndarray:
    """The ``channel`` of ``x`` (samples, or samples by channels): left or right (a mono
    file's one channel for both), or all of them mixed down (mono, or None)."""
    if channel in ("left", "right"):
        return stereo(x)[:, 0 if channel == "left" else 1]
    return mono(x)


def feeds(args: argparse.Namespace) -> None:
    """One WAV per source that plays: its file (its signal, what ``--input`` gives it, or
    the scene's recording), at the scene's sample rate, the channel the source names or
    mixed down to mono. Each file is brought to one gated RMS (FEED_RMS; its channels
    mixed down), and the channels taken out of it keep that gain, so that its stereo
    balance stays and ``auralize`` sets the balance between the sources from their sound
    powers; sources without one are listed as silent."""
    scene = scene_of(args)
    rate = scene.simulation.sample_rate
    out = work_of(args) / "feeds"
    table: dict[str, dict] = {}
    silent: list[str] = []
    length = 0
    free = free_of(scene)
    read_in: dict[Path, tuple[np.ndarray, float]] = {}
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
        if signal not in read_in:
            x, r = decoded(signal)
            x = resampled(x, r, rate)
            rms = gated_rms(x, rate)
            if rms <= 0.0:
                raise SystemExit(f"error: {signal} is silent")
            read_in[signal] = (x, FEED_RMS / rms)
        x, gain = read_in[signal]
        y = channel_of(x, channel) * gain
        write(out / f"{name}.wav", y, rate)
        length = max(length, len(y))
        table[name] = {"signal": str(signal), "channel": channel, "length_s": len(y) / rate}
        which = "" if channel is None else f", {CHANNEL_WORDS[channel]}"
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
        {"rate": rate, "length_s": length / rate, "sources": table, "silent": silent},
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


def render(args: argparse.Namespace, demo: Demo | None = None) -> None:
    """Every listener's mix (``--format``, binaural by default) with one gain for all, the
    loudest peak at -1 dBFS; ``auralize`` plays each feed at its source's sound power.
    The arrival time counts the sources that play, each at its sound power. When
    something moves, :func:`render_moving` makes the mixes instead. ``demo``, this one by
    default, is the demo that runs it, whose commands the messages give."""
    moving = moving_of(args)
    if moving:
        render_moving(args, moving, demo)
        return
    table = feeds_table(args, demo)
    entries = responses_of(args, demo)
    simulation = simulations(args)[0]
    feeds = {name: work_of(args) / "feeds" / f"{name}.wav" for name in table["sources"]}
    powers = {s.name: s.sound_power_db for s in scene_of(args).sources if s.name in feeds}
    mixes = [
        pipeline.Mix(name, simulation, name, feeds, weights=powers)
        for name in listeners_of(args, entries)
    ]
    if not mixes:
        raise SystemExit("error: no listener to render; the scene's listeners are all left out")
    pipeline.render_mixes(mixes, work_of(args) / "seats", args)


def format_of(args: argparse.Namespace, scene: Scene):
    """The format the moving mixes take: the scene's output of ``--format`` (as its
    simulation renders it), through ``--hrtf`` when given."""
    from acoustic_spaces.scene import Output
    from acoustic_spaces.spatial import from_output

    output = next((o for o in scene.outputs if o.format == args.format), None)
    if output is None:
        output = Output.model_validate(args.format)
    if args.hrtf and output.format == "binaural":
        output = output.model_copy(update={"hrtf": Path(args.hrtf)})
    return from_output(output)


def moving_done(
    args: argparse.Namespace, folder: Path, moving: list[str], names: list[str]
) -> bool:
    """Whether the moving mixes in ``folder`` are there for this run: the same walks,
    listeners and format, newer than the scene and the feeds."""
    path = folder / pipeline.SEATS
    if getattr(args, "again", False) or not path.exists():
        return False
    seats = pipeline.read_json(path)
    same = (
        seats.get("moving") == moving
        and seats.get("label") == pipeline.output_label(args)
        and sorted(seats.get("seats", {})) == sorted(names)
    )
    made = [Path(args.scene), work_of(args) / "feeds" / FEEDS]
    return same and all(path.stat().st_mtime >= p.stat().st_mtime for p in made if p.exists())


def levels_along(movement, source, listener: str) -> list[float | None]:
    """The A-weighted level (dB) of ``source`` (a scene source) at ``listener`` at each
    moment of ``movement``, as simulated; None at a moment whose response is too short
    for its decay."""
    from acoustic_spaces.parameters import a_weighted

    power = source.power_db("octave")
    levels: list[float | None] = []
    for k in range(len(movement.times)):
        response = movement.result[movement.key(source.name, listener, k)]
        level = float(a_weighted(response.levels(power)))
        levels.append(round(level, 2) if math.isfinite(level) else None)
    return levels


def render_moving(args: argparse.Namespace, moving: list[str], demo: Demo | None = None) -> None:
    """Every listener's mix with ``moving`` walking, all on one clock: the scene simulated
    with every mover at each moment of its walk
    (:func:`~acoustic_spaces.walkthrough.simulate_movement`), each feed played through
    it at its source's sound power, one gain for all (the loudest peak at -1 dBFS), into
    ``moving/`` with seats.json; a walk with Doppler, which moves alone, with each early
    path through its changing delay. That also keeps the moments and, at each, the
    A-weighted level of each source that plays at every listener, for the pictures; the
    arrival time (``arrival_s``) is left out where something moves, as it changes along
    the way, and the tour shifts no mix by it (``lag_s`` 0), so that the stops keep one
    clock."""
    from acoustic_spaces.room import Room
    from acoustic_spaces.walkthrough import render_movement, simulate_movement

    table = feeds_table(args, demo)
    names = listeners_of(args, responses_of(args, demo))
    if not names:
        raise SystemExit("error: no listener to render; the scene's listeners are all left out")
    work = work_of(args)
    folder = work / MOVING
    if moving_done(args, folder, moving, names):
        print(f"{folder}: there already (--again mixes them again)")
        return
    scene = moving_scene(args)
    room = Room.from_scene(scene)
    rate = int(scene.simulation.sample_rate)
    feeds = {name: read(work / "feeds" / f"{name}.wav") for name in table["sources"]}
    playing = [s for s in scene.sources if s.name in feeds]
    fmt = format_of(args, scene)
    print(f"simulating the movement of {', '.join(moving)}, on one clock")
    movement = simulate_movement(room, progress=print)
    with_doppler = ", each early path through its changing delay" if movement.doppler else ""
    print(f"mixing {len(names)} listeners through {len(movement.times)} moments{with_doppler}")
    mixes = render_movement(movement, feeds, fmt, sample_rate=rate)
    times, places = movement.times, movement.places
    levels = {name: {s.name: levels_along(movement, s, name) for s in playing} for name in names}
    peak = max(float(abs(mixes[name]).max(initial=0.0)) for name in names)
    gain = 10.0 ** (-1.0 / 20.0) / peak if peak > 0 else 1.0
    weights = {s.name: s.sound_power_db for s in playing}
    sources_move = any(s.name in moving for s in playing)
    entries = responses_of(args, demo)
    info = {}
    for name in names:
        x = mixes[name] * gain
        write(folder / f"{name}.{args.format}.wav", x, rate)
        moves = sources_move or name in moving
        # Where nothing moves, the sound arrives as in the still simulation. The tour
        # shifts none of the mixes, as they keep the walks' one clock.
        arrival = None if moves else pipeline.arrival_s(results_of(args), entries, name, weights)
        info[name] = {
            "listener": name,
            "lag_s": 0.0,
            "arrival_s": arrival,
            "rms_db": level_db(x, rate),
            "loudness_db": loudness_db(x, rate),
            "length_s": len(x) / rate,
            "moves": moves,
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
        "seats": info,
        "movement": {"times": [float(t) for t in times], "levels_a_db": levels},
    }
    pipeline.write_json(folder / pipeline.SEATS, seats)
    print(f"wrote {folder}: {len(names)} mixes, {places} places along the walks")


# --------------------------------------------------------------------------- the video


def tour_of(
    args: argparse.Namespace, seats: dict, length_s: float, demo: Demo | None = None
) -> list[pipeline.TourStop]:
    """The stops: each listener for ``--seconds`` while the piece runs on; a piece too
    short for that shares its length out."""
    entries = responses_of(args, demo)
    names = [name for name in listeners_of(args, entries) if name in seats["seats"]]
    if not names:
        run = (demo or DEMO).command("render", args)
        raise SystemExit(f"error: no mix to tour; run first: {run}")
    each = min(float(args.seconds), length_s / len(names))
    if each < float(args.seconds) - 1e-6:
        print(f"the piece lasts {length_s:.0f} s: each listener is heard {each:.1f} s")
    gains = pipeline.gains_of(seats, names)
    return [
        pipeline.TourStop(name, k * each, k * each, (k + 1) * each, gains[name])
        for k, name in enumerate(names)
    ]


def video(args: argparse.Namespace, demo: Demo | None = None) -> None:
    """tour.wav, one still per stop, the thumbnail, and tour.mp4: the stills with a
    scrolling spectrogram in the panel at their foot (ffmpeg, unless ``--no-mp4``). When
    something moves, the video is a moving picture instead: the walkers go along their
    walks as the piece plays, with a still of each stop. ``demo``, this one by default,
    is the demo that runs it: its commands and its settings go in the messages and on
    the stills."""
    demo = demo or DEMO
    from acoustic_spaces.demos import pictures, scene_pictures
    from acoustic_spaces.room import Room, outline_edges
    from acoustic_spaces.walkthrough import knots, movers

    work = work_of(args)
    table = feeds_table(args, demo)
    moving = moving_of(args)
    folder = work / (MOVING if moving else "seats")
    seats = pipeline.seats_of(folder, demo.command("render", args))
    if seats.get("moving", []) != moving:
        raise SystemExit(
            f"error: the mixes in {folder} are of other walks; run first: "
            f"{demo.command('render', args)}"
        )
    stops = tour_of(args, seats, float(table["length_s"]), demo)
    audio = pipeline.tour_audio(
        stops,
        folder,
        seats,
        crossfade_s=min(CROSSFADE_S, stops[0].length / 2),
        fade_out_s=min(FADE_OUT_S, stops[-1].length / 2),
    )
    write(work / "tour.wav", audio, seats["rate"])
    print(f"wrote {work / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    scene = moving_scene(args)
    data = scene.to_data()
    report = pipeline.read_json(results_of(args) / pipeline.PARAMETERS)
    outline = outline_edges(Room.from_scene(scene))[0]
    drawing = scene_pictures.Pictures(
        data,
        outline,
        report,
        seats["seats"],
        stops,
        playing=list(table["sources"]),
        credit=args.credit or plays_line(table, free_of(scene)),
        tool=pictures.settings_lines(demo, args, data, report),
        fmt=args.format,
        paths={(kind, item.name): knots(item) for kind, item in movers(scene)},
        movement=seats.get("movement"),
    )
    pictures.save(drawing.thumbnail(), work / "thumbnail.png")
    if moving:
        pipeline.animation(
            work,
            len(audio) / seats["rate"],
            drawing.frame,
            work / "tour.wav",
            stills=[(stop.key, stop.at + min(0.5, stop.length / 2)) for stop in stops],
            no_mp4=args.no_mp4,
        )
        return
    pipeline.video(
        work,
        [(stop.key, stop.length) for stop in stops],
        drawing.still,
        work / "tour.wav",
        no_mp4=args.no_mp4,
    )


def plays_line(table: dict, free: recordings.Recording | None = None) -> str:
    """What the sources play, for the stills when ``--credit`` says nothing: the credit
    the free recording's licence asks for, when one plays, and each other file by
    source."""
    kept = str(recordings.file_of(free)) if free is not None else None
    parts = (
        [free.credit]
        if free is not None and any(v["signal"] == kept for v in table["sources"].values())
        else []
    )
    others = [
        f"{name}: {Path(v['signal']).name}"
        for name, v in table["sources"].items()
        if v["signal"] != kept
    ]
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
        Step("video", video, "tour.wav, the stills and tour.mp4 in the work folder"),
    ],
    options=[
        Option(
            "scene",
            "the scene to demo: its sources play their signals",
            None,
            "file",
            filter="Scene files (*.yaml *.yml)",
        ),
        Option(
            "seconds",
            "how long each listener is heard (s); a piece too short for every listener "
            "shares its length out",
            20.0,
            "float",
            steps=("video",),
        ),
        dataclasses.replace(
            listeners_option(),
            help="only these listeners, in this order, comma separated (default: every "
            "listener in use, in the scene's order): the demo mixes those alone and its "
            "video visits them in turn; the simulation keeps every listener, so another "
            "choice needs no new one",
            aliases=("--tour",),
        ),
        Option(
            "credit",
            "a line on the stills about the recordings: who made them, under which licence",
            None,
            steps=("video",),
        ),
        Option(
            "moving",
            "the sources and listeners that move along their walks in this run, by name, "
            "comma separated, every other walker standing where it starts; none keeps them "
            "all still (default: those whose walk is on in the scene)",
            None,
            "movers",
            steps=("render", "video"),
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
    plays=plays,
    moving=moving_of,
    listeners=lambda args: listeners_in(args.scene),
)

if __name__ == "__main__":
    sys.exit(DEMO.main())
