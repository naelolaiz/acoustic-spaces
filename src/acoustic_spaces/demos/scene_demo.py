"""A demo of any scene: what its sources play, heard at each of its listeners, as a video.

Each source plays the WAV file its ``signal`` names (set on the source's page in the
editor, **Plays**, or in the scene file), or the one ``--input SOURCE=WAV`` gives it
instead, dry: mixed down to mono and brought to the same gated RMS, so that its sound
power in the scene (``level_db``) sets how loud it is against the others, as the
simulation says and not as the recordings happen to be. Sources without one stay
silent. Every listener in use is heard in turn, in the scene's order (or ``--tour``),
for ``--seconds`` each while the piece runs on, with one gain for all the stops and a
crossfade between them, binaural through the HRTFs of the scene's binaural output (or
``--hrtf``, or a spherical head model when neither names any). The stills show the
scene from above with the listener of the moment, the level there, how long the sound
takes to arrive and T30 at mid frequencies, and say how the sound was made.

    acoustic-spaces demo scene.yaml              # every step, into demo/ beside the scene
    acoustic-spaces demo scene.yaml feeds        # one step
    acoustic-spaces demo scene.yaml --hrtf me.sofa --seconds 15 --tour front,back
    acoustic-spaces demo scene.yaml --input talker=other_take.wav

The editor runs it from **Simulation > Run demo…** as *This scene*. The video needs
ffmpeg on the PATH and Pillow (the ``viz`` extra). The five demos of the examples are
scripts of their own, with their recordings; this one takes any scene.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import sys
from pathlib import Path

from acoustic_spaces.demos import (
    SIMULATE,
    Demo,
    Option,
    Step,
    common_options,
    input_option,
    pipeline,
)
from acoustic_spaces.demos.audio import gated_rms, mono, read, resampled, write
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
CROSSFADE_S = 1.0
FADE_OUT_S = 3.0


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


def plays(args: argparse.Namespace) -> dict[str, str]:
    """Every source in use and what it plays: the name of its signal, or silent."""
    found = _current(getattr(args, "scene", None))
    if found is None:
        return {}
    return {
        s.name: (s.signal.name if s.signal is not None else "silent")
        for s in found.sources
        if s.in_use
    }


def signals_of(args: argparse.Namespace, scene: Scene) -> dict[str, Path | None]:
    """What each source in use plays: the file ``--input`` gives it, else its signal
    (None: silent)."""
    inputs = getattr(args, "inputs", None) or {}
    return {s.name: inputs.get(s.name, s.signal) for s in scene.sources if s.in_use}


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


def results_of(args: argparse.Namespace) -> Path:
    return work_of(args) / "results"


def scene_of(args: argparse.Namespace) -> Scene:
    """The scene, its sources' signals resolved against its folder."""
    return Scene.from_yaml(args.scene)


def simulations(args: argparse.Namespace) -> list[pipeline.Simulation]:
    return [pipeline.Simulation(Path(args.scene), results_of(args), args.format)]


def responses_of(args: argparse.Namespace) -> list[dict]:
    return pipeline.responses_of(simulations(args)[0], args, DEMO.command("simulate", args))


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
            "editor, signal: in the scene file, or --input SOURCE=WAV)"
        )
    for name, signal in signals.items():
        if not Path(signal).exists():
            return f"{name} plays {signal}, which is not there"
    return None


# --------------------------------------------------------------------------- the feeds


def feeds_table(args: argparse.Namespace) -> dict:
    path = work_of(args) / "feeds" / FEEDS
    if not path.exists():
        raise SystemExit(f"error: no feeds yet; run first: {DEMO.command('feeds', args)}")
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
    wanted = {k: str(v) for k, v in signals_of(args, scene).items() if v is not None}
    if wanted != {k: v["signal"] for k, v in table["sources"].items()}:
        return False
    files = [Path(args.scene), *(Path(v) for v in wanted.values())]
    return all(not f.exists() or f.stat().st_mtime <= made for f in files)


def feeds(args: argparse.Namespace) -> None:
    """One WAV per source that plays: its signal (or what ``--input`` gives it), mono,
    at the scene's sample rate and at one gated RMS (FEED_RMS), so that ``auralize`` sets
    the balance from the sources' sound powers; sources without one are listed as
    silent."""
    scene = scene_of(args)
    rate = scene.simulation.sample_rate
    out = work_of(args) / "feeds"
    table: dict[str, dict] = {}
    silent: list[str] = []
    length = 0
    for name, signal in signals_of(args, scene).items():
        if signal is None:
            silent.append(name)
            continue
        signal = Path(signal)
        if not signal.exists():
            raise SystemExit(f"error: {name} plays {signal}, which is not there")
        x, r = read(signal)
        x = resampled(mono(x), r, rate)
        rms = gated_rms(x, rate)
        if rms <= 0.0:
            raise SystemExit(f"error: {signal} is silent")
        x = x * (FEED_RMS / rms)
        write(out / f"{name}.wav", x, rate)
        length = max(length, len(x))
        table[name] = {"signal": str(signal), "length_s": len(x) / rate}
        print(f"{name:>20}: {signal.name} ({len(x) / rate:.1f} s)")
    if not table:
        raise SystemExit(
            f"error: no source of {args.scene} plays anything; give one a WAV file: Plays on "
            "the source's page in the editor, signal: in the scene file, or --input SOURCE=WAV"
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
    """The listeners of the tour: ``--tour``, or every listener in use that the results
    have, in the scene's order."""
    simulated = {e["listener"] for e in entries}
    if args.tour:
        names = [name.strip() for name in args.tour.split(",") if name.strip()]
        missing = [name for name in names if name not in simulated]
        if missing:
            raise SystemExit(
                f"error: no listener {missing[0]!r} in the results; they have "
                + ", ".join(sorted(simulated))
            )
        return names
    scene = scene_of(args)
    return [lis.name for lis in scene.listeners if lis.in_use and lis.name in simulated]


def render(args: argparse.Namespace) -> None:
    """Every listener's mix (``--format``, binaural by default) with one gain for all, the
    loudest peak at -1 dBFS; ``auralize`` plays each feed at its source's sound power.
    The arrival time counts the sources that play, each at its sound power."""
    table = feeds_table(args)
    entries = responses_of(args)
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


# --------------------------------------------------------------------------- the video


def tour_of(args: argparse.Namespace, seats: dict, length_s: float) -> list[pipeline.TourStop]:
    """The stops: each listener for ``--seconds`` while the piece runs on; a piece too
    short for that shares its length out."""
    names = [name for name in listeners_of(args, responses_of(args)) if name in seats["seats"]]
    if not names:
        raise SystemExit(f"error: no mix to tour; run first: {DEMO.command('render', args)}")
    each = min(float(args.seconds), length_s / len(names))
    if each < float(args.seconds) - 1e-6:
        print(f"the piece lasts {length_s:.0f} s: each listener is heard {each:.1f} s")
    gains = pipeline.gains_of(seats, names)
    return [
        pipeline.TourStop(name, k * each, k * each, (k + 1) * each, gains[name])
        for k, name in enumerate(names)
    ]


def video(args: argparse.Namespace) -> None:
    """tour.wav, one still per stop, the thumbnail, and tour.mp4: the stills with a
    scrolling spectrogram in the panel at their foot (ffmpeg, unless ``--no-mp4``)."""
    from acoustic_spaces.demos import pictures, scene_pictures
    from acoustic_spaces.room import Room, outline_edges

    work = work_of(args)
    table = feeds_table(args)
    seats = pipeline.seats_of(work / "seats", DEMO.command("render", args))
    stops = tour_of(args, seats, float(table["length_s"]))
    audio = pipeline.tour_audio(
        stops,
        work / "seats",
        seats,
        crossfade_s=min(CROSSFADE_S, stops[0].length / 2),
        fade_out_s=min(FADE_OUT_S, stops[-1].length / 2),
    )
    write(work / "tour.wav", audio, seats["rate"])
    print(f"wrote {work / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    scene = scene_of(args)
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
        credit=args.credit or plays_line(table),
        tool=pictures.settings_lines(DEMO, args, data, report),
        fmt=args.format,
    )
    pictures.save(drawing.thumbnail(), work / "thumbnail.png")
    pipeline.video(
        work,
        [(stop.key, stop.length) for stop in stops],
        drawing.still,
        work / "tour.wav",
        no_mp4=args.no_mp4,
    )


def plays_line(table: dict) -> str:
    """What each source plays, for the stills when ``--credit`` says nothing."""
    plays = [f"{name}: {Path(v['signal']).name}" for name, v in table["sources"].items()]
    return "Playing " + ", ".join(plays)


DEMO = Demo(
    name="scene",
    example="",
    title=N_("This scene"),
    description=__doc__,
    script=Path(__file__),
    steps=[
        Step(
            "feeds",
            feeds,
            "one WAV per source that plays: its signal, mono, at one gated RMS",
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
        Option(
            "tour",
            "only these listeners, in this order, comma separated (default: every listener "
            "in use, in the scene's order)",
            None,
            steps=("render", "video"),
        ),
        Option(
            "credit",
            "a line on the stills about the recordings: who made them, under which licence",
            None,
            steps=("video",),
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
)

if __name__ == "__main__":
    sys.exit(DEMO.main())
