"""A song played on the lakeside concert example's stage, heard in the audience, behind
the stage, from a boat and in the village across the lake, and a video that tours them.

The stage's sound system, two line arrays and subwoofers, plays a stereo recording
(--song): by default
"Spicy Funk Cake" by the Rhythmussportgruppe, a nine-piece funk band, recorded for the
German Acoustical Society (DEGA) at Hochschule Düsseldorf and published under CC BY 4.0
(https://doi.org/10.5281/zenodo.3601032), or a song of your own (--music, with --bpm its
tempo). At a bedroom window across the
lake the tour plays the song in the evening's weather, which bends the sound down to
the water, then in still air, where it goes straight, then in the evening's again.
Run it from the folder of the lakeside concert example (`acoustic-spaces examples
--copy . lakeside_concert && cd lakeside_concert`), in the Python environment
acoustic-spaces is installed in, with ffmpeg on the PATH:

    python lakeside_demo.py music     # download the song, a signal per loudspeaker
    python lakeside_demo.py scenes    # scene_demo.yaml, and scene_still.yaml in still air
    python lakeside_demo.py simulate  # both scenes: results_demo and results_still
    python lakeside_demo.py render    # every listener in both, binaural
    python lakeside_demo.py video     # demo/lakeside_spicy_funk_cake_binaural.mp4

or `python lakeside_demo.py all`, which runs them in turn and leaves out what is there
already (the song, the scenes, the simulations). The scenes and their simulations go
beside the script (scene_demo.yaml and scene_still.yaml, results_demo and
results_still, a folder in each for each output, as results_demo/binaural); the song
and its signals go to ./demo/music (--work), and each run's mixes and stills to a
folder in ./demo named after its settings (what it plays, the places, the output:
demo/lakeside_spicy_funk_cake_binaural/), with the video, its audio and its thumbnail
beside it, so that a run with other settings writes over none of them. Use
headphones; `--hrtf me.sofa` plays through your own or measured HRTFs instead of a
spherical head (the simulations take it too). `--input subs=bass.wav` has a
loudspeaker (main_left, main_right or subs) play a file of yours instead of its part of
the song, at the level of that part. `--music song.wav --bpm 120` plays a song of your
own (WAV, or anything ffmpeg decodes) in place of --song: nothing is downloaded, the
tour moves every 8 bars of the tempo given, and the video credits the file by its name
unless `--credit` says what to show. The editor runs the same steps from Simulation >
Run demo…

From the audience to the far shore the sound drops by more than 20 dB(A), too much to
hear the far places in a video. So each stop of the tour is brought to the same loudness
(K-weighted, as loudness meters weigh it) as the audience's, and its caption says how
much quieter it really is. The window in still air keeps the gain of the window in the
evening's weather, so the difference the weather makes is heard as the simulation gives
it. Each place's mix is shifted earlier by the time the sound takes to reach it (1.6 s
across the lake), so that the beat carries on across the cuts. `--listeners
bedroom_window,audience` mixes those places alone, in both weathers, and the video tours
them in that order, each place with its stops of the tour (the window in the evening, in
still air and in the evening again), and back to the first at the end; `--tour` gives a
tour of your own, which --listeners then only thins out. `--sources subs` plays the
subwoofers alone (`main_left`, `main_right` and `subs`, or `PA` for all three, which is
the default): the simulations keep every loudspeaker, so another pick needs no new ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from acoustic_spaces import recordings
from acoustic_spaces.demos import (
    SIMULATE,
    Demo,
    Option,
    Source,
    Step,
    chosen_order,
    common_options,
    input_option,
    listener_names,
    listeners_option,
    music,
    name_part,
    picked,
    pipeline,
    playing,
    product,
    run_folder,
    sibling,
    sources_options,
)
from acoustic_spaces.demos.audio import channels_rms, gated_rms, mono, read, stereo, write
from acoustic_spaces.demos.scene_demo import FEED_LEVELS, FEED_RMS

#: The songs the demo can play (``--song``): the free songs of
#: :mod:`acoustic_spaces.recordings`, each with the files it is made of (one here, a
#: member of a zip archive read by HTTP range requests or a file of its own, as
#: :mod:`acoustic_spaces.demos.music` fetches them), the MB they take to download, its
#: tempo (beats a minute; the tour moves on every 8 bars) and the credit its licence
#: asks for.
SONGS = {
    item.name: {
        "files": list(item.files),
        "download_mb": item.download_mb,
        "bpm": item.bpm,
        "credit": item.credit,
    }
    for item in recordings.recordings()
    if item.kind == "song"
}
#: The rate the songs are brought to, that of the simulation's responses.
RATE = 48000
#: What ``music`` keeps of a song in song.json.
SONG_KEPT = ("name", "bpm", "credit")


#: The stage's sound system: the scene's sources in this group, left out of its own
#: simulations, and the channel of the song each plays (the subwoofers both, in mono).
PA = "PA"
CHANNELS = {"main_left": 0, "main_right": 1, "subs": "mono"}
#: Each loudspeaker in words, for the stills of a run that plays some of them.
LOUDSPEAKERS = {
    "main_left": "the left line array",
    "main_right": "the right line array",
    "subs": "the subwoofers",
}
#: The scene's listener behind the stage, as far upwind as the park reaches, left out of
#: its own simulations.
UPWIND = "behind_stage"
#: How long the responses run past the latest direct sound, the far shore's, 1.6 s late:
#: outdoors the simulation counts ``ir_length_s`` from there (5.2 s in all).
IR_LENGTH_S = 3.5

#: The default tour: listener, weather ("evening" or "still") and how many phrases it
#: lasts; the last one, the closing stop, runs to the end of the song. ``--listeners``
#: tours its places in its order, each with its stops here but the closing one, and
#: closes at the first (:func:`tour_steps`).
TOUR = [
    ("audience", "evening", 2),
    ("behind_stage", "evening", 1),
    ("boat", "evening", 1),
    ("garden", "evening", 1),
    ("bedroom_window", "evening", 1),
    ("bedroom_window", "still", 1),
    ("bedroom_window", "evening", 1),
    ("audience", "evening", 0),
]
TITLES = {
    "audience": ("In the audience", "on the lawn, 30 m from the stage"),
    "behind_stage": ("Behind the stage", "140 m back in the park, upwind"),
    "boat": ("From a boat", "in the middle of the lake, 330 m downwind"),
    "garden": ("A garden across the lake", "in the village, 550 m away"),
    "bedroom_window": ("At a bedroom window", "5 m up a house in the village, 570 m away"),
}
CROSSFADE_S = 1.0
FADE_OUT_S = 4.0

# --------------------------------------------------------------------------- the music


def song_of(args) -> dict:
    """The song: ``--song``'s preset, or the song of your own ``--music`` gives, named
    after its file, at the tempo of ``--bpm``, with the credit of ``--credit`` or the
    file's name."""
    if args.music is None:
        return SONGS[args.song] | {"name": args.song}
    return {
        "name": args.music.stem,
        "files": args.music,
        "bpm": args.bpm,
        "credit": args.credit or f"Music: {args.music.name}",
    }


def phrase_s(song: dict) -> float:
    """8 bars of ``song``, s."""
    return music.phrase_s(song)


def music_folder(args) -> Path:
    return args.work / "music"


def feeds_folder(args) -> Path:
    """Where the signals of the song a run plays go, with ``song.json``: a folder of the
    song's own in the music folder, so that runs of other songs keep theirs."""
    return music_folder(args) / name_part(song_of(args)["name"])


def song_path(args) -> Path:
    return music_folder(args) / f"{song_of(args)['name']}.wav"


def music_done(args) -> bool:
    """Whether the song and each loudspeaker's signal are there, made from the song the
    run plays (its name, tempo and credit) and newer than a song of your own."""
    folder = feeds_folder(args)
    if not song_path(args).exists():
        return False
    if not all((folder / f"{name}.wav").exists() for name in CHANNELS):
        return False
    path = folder / "song.json"
    if not path.exists():
        return False
    if args.music is not None and Path(args.music).exists():
        if path.stat().st_mtime < Path(args.music).stat().st_mtime:
            return False
    song = pipeline.read_json(path)
    kept = {k: v for k, v in song_of(args).items() if k in SONG_KEPT}
    # Signals levelled otherwise (the song's raw channels, before 4.1.0) are made again.
    return song.get("levels") == FEED_LEVELS and all(song.get(k) == v for k, v in kept.items())


def music_download(args) -> str | None:
    """What ``music`` downloads: ``--song``, nothing for a song of your own."""
    if args.music is not None:
        return None
    return f"the song {args.song} ({SONGS[args.song]['download_mb']} MB)"


def music_step(args) -> None:
    """Download ``--song`` (one member of a zip archive, or the file itself), or read the
    song of your own ``--music`` names, bring it to 48 kHz stereo, and write what each
    loudspeaker plays (CHANNELS), each at the level its sound power sets: the left and
    right channels share one gain that brings their mean square to FEED_RMS, and the
    subwoofers' mixdown, which two unlike channels lower, is brought to it too. A song
    downloaded before is kept."""
    song = song_of(args)
    music.write_tracks({song["name"]: song["files"]}, music_folder(args), rate=RATE, channels=2)
    folder = feeds_folder(args)
    x, rate = read(song_path(args))
    x = stereo(x)
    pair = channels_rms(x, rate)
    for name, channel in CHANNELS.items():
        y = mono(x) if channel == "mono" else x[:, channel]
        level = gated_rms(y, rate) if channel == "mono" else pair
        if level <= 0.0:
            raise SystemExit(f"error: {song['name']} is silent")
        write(folder / f"{name}.wav", y * (FEED_RMS / level), rate)
    kept = {k: v for k, v in song.items() if k in SONG_KEPT}
    pipeline.write_json(folder / "song.json", kept | {"levels": FEED_LEVELS})
    print(f"wrote {', '.join(CHANNELS)} ({len(x) / rate:.0f} s of {song['name']}) to {folder}")


def sources(args) -> list[Source]:
    """The loudspeakers a run can play (``--sources``), all of them by default: the
    stage's sound system, its PA group."""
    return [Source(name, PA) for name in CHANNELS]


def what_plays(names) -> str | None:
    """What the stills say of a run that plays only the loudspeakers ``names`` of the
    sound system; None when they all play."""
    if set(CHANNELS) <= set(names):
        return None
    return "Playing only " + " and ".join(LOUDSPEAKERS[name] for name in names)


def plays(args) -> dict[str, str]:
    """What each loudspeaker plays unless ``--input`` gives it a file: its channel of the
    song."""
    channels = {0: "left channel", 1: "right channel", "mono": "both channels in mono"}
    song = song_of(args)["name"]
    return {name: f"{song}, {channels[channel]}" for name, channel in CHANNELS.items()}


def played(args) -> dict:
    """The song ``music`` wrote the loudspeakers' signals from, for the song the run
    plays."""
    path = feeds_folder(args) / "song.json"
    if not path.exists():
        raise SystemExit(f"error: no song yet; run first: {DEMO.command('music', args)}")
    return {k: v for k, v in pipeline.read_json(path).items() if k in SONG_KEPT}


# --------------------------------------------------------------------------- the scenes


def scenes(args) -> None:
    """``--scene`` (scene_demo.yaml) from ``--from`` (scene.yaml), with the stage's
    sound system (its PA group) in place of the example's one source, its listener
    behind the stage taken in and responses long enough for the far shore; and
    ``--still`` (scene_still.yaml), the same in still air."""
    scene = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    scene["name"] = scene["name"] + " (demo)"
    scene["sources"] = [
        {k: v for k, v in s.items() if k != "enabled"}
        for s in scene["sources"]
        if s.get("group") == PA
    ]
    names = [s["name"] for s in scene["sources"]]
    missing = [name for name in CHANNELS if name not in names]
    if UPWIND not in {item["name"] for item in scene["listeners"]}:
        missing.append(UPWIND)
    if missing:
        raise SystemExit(f"error: {args.source} has no {', '.join(missing)}")
    scene["listeners"] = [
        {k: v for k, v in item.items() if k != "enabled" or item["name"] != UPWIND}
        for item in scene["listeners"]
    ]
    scene.setdefault("simulation", {})["ir_length_s"] = IR_LENGTH_S
    still = dict(scene, name=scene["name"].replace("(demo)", "(demo, still air)"))
    still.pop("atmosphere", None)
    for path, data, what in (
        (args.scene, scene, "with its sound system"),
        (args.still, still, "in still air, with its sound system"),
    ):
        pipeline.write_scene(
            path,
            data,
            f"# The lakeside concert example {what}, written by lakeside_demo.py.\n",
            args.source.parent,
            made_from=args.source,
        )
    print(f"wrote {args.scene} and {args.still}; now:")
    for simulation in simulations(args):
        print(f"  {pipeline.simulate_command(simulation, args.hrtf)}")


def weathers(args) -> dict[str, pipeline.Simulation]:
    """Each weather's scene and the folder its simulation writes for the run's output
    (:func:`~acoustic_spaces.demos.pipeline.results_for`)."""
    return {
        "evening": pipeline.Simulation(
            args.scene, pipeline.results_for(args.results, args), args.format
        ),
        "still": pipeline.Simulation(
            args.still, pipeline.results_for(args.still_results, args), args.format
        ),
    }


def simulations(args) -> list[pipeline.Simulation]:
    """Both weathers, simulated one after the other."""
    return list(weathers(args).values())


def responses_of(args, weather: str) -> list[dict]:
    return pipeline.responses_of(weathers(args)[weather], args, DEMO.command("scenes", args))


# --------------------------------------------------------------------------- the mixes


def render(args) -> None:
    """Every listener's mix in both weathers (``--format``, binaural by default), with one
    gain for all: the loudest peak at -1 dBFS; and each one's level and delay. Only the
    loudspeakers ``--sources`` picks play, and only their sound counts in when it
    arrives (the weights, all alike)."""
    feeds = {name: feeds_folder(args) / f"{name}.wav" for name in CHANNELS}
    weights = {name: 0.0 for name in CHANNELS}
    mixes = []
    for weather, simulation in weathers(args).items():
        simulated = sorted({e["listener"] for e in responses_of(args, weather)})
        for listener in picked(args, simulated):
            mixes.append(
                pipeline.Mix(
                    f"{listener}.{weather}",
                    simulation,
                    listener,
                    feeds,
                    weights,
                    info={"weather": weather},
                )
            )
    pipeline.render_mixes(
        mixes,
        run_folder(args) / "seats",
        args,
        first=DEMO.command("scenes", args),
        inputs=args.inputs,
    )


def tour_steps(names) -> list[tuple[str, str, float]]:
    """The tour of the places ``names``, in their order: each one's stops of the default
    tour (TOUR) but its closing stop, one in the evening's weather for a place it does
    not visit, then the closing stop at the first. The default tour for its own places
    in its own order."""
    *stops, (_, _, closing) = TOUR
    steps = []
    for name in names:
        own = [step for step in stops if step[0] == name]
        steps += own or [(name, "evening", 1.0)]
    return [*steps, (names[0], "evening", closing)]


def tour_of(args) -> list[tuple[str, str, float, float]]:
    """(listener, weather, start, end) in the song's time; the stops end on the song's
    phrase boundaries (8 bars). The places ``--listeners`` names, in its order
    (:func:`tour_steps`), or with ``--tour``, its stops at those places; stops at one
    place in one weather in a row are joined."""
    if args.tour:
        steps = []
        for part in args.tour.split(","):
            fields = part.split(":")
            weather = fields[1] if len(fields) > 1 and fields[1] else "evening"
            steps.append((fields[0], weather, float(fields[2]) if len(fields) > 2 else 1.0))
        kept = set(picked(args, [listener for listener, _, _ in steps]))
        steps = [step for step in steps if step[0] in kept]
    elif listener_names(args.listeners) is None:
        steps = TOUR
    else:
        names = chosen_order(args, listeners(args))
        steps = tour_steps(names) if names else []
    if not steps:
        raise SystemExit("error: the tour visits none of the listeners --listeners names")
    song = played(args)
    x, rate = read(song_path(args))
    tour = music.phrase_tour(
        [((listener, weather), phrases) for listener, weather, phrases in steps],
        len(x) / rate,
        phrase_s(song),
        last_phrases=args.last_phrases,
    )
    stops = [(listener, weather, start, stop) for (listener, weather), start, stop in tour]
    return pipeline.picked_tour(stops, lambda stop: True)


def listeners(args) -> list[str]:
    """The places the scenes step puts in the demo's scenes: its listeners in use, and
    the one behind the stage, which the example leaves out of its own simulations; in
    the order the default tour first visits them, then any other in the example's
    order."""
    data = yaml.safe_load(args.source.read_text(encoding="utf-8")) or {}
    names = [
        item["name"]
        for item in data.get("listeners") or []
        if item.get("enabled", True) is not False or item["name"] == UPWIND
    ]
    toured = [name for name in dict.fromkeys(step[0] for step in TOUR) if name in names]
    return toured + [name for name in names if name not in toured]


def gains_of(seats: dict, tour) -> dict[str, float]:
    """The gain (dB) each stop plays at: the loudest place's loudness everywhere, but a
    place in still air at the gain of the same place in the evening's weather. Loudness
    as meters weigh it (K), not the A weighting of the levels: the far places and the
    back of the stage keep little but bass, which the A weighting all but leaves out, so
    matching levels would turn their bass up far past the audience's."""
    evening = {f"{listener}.{weather}": f"{listener}.evening" for listener, weather, _, _ in tour}
    known = {key: twin if twin in seats["seats"] else key for key, twin in evening.items()}
    gains = pipeline.gains_of(seats, sorted(set(known.values()) | set(seats["seats"])))
    return {key: gains[twin] for key, twin in known.items()}


def playing_sources(scene: dict, names) -> list[dict]:
    """The sources of the scene ``scene`` (as read) among ``names``, the loudspeakers
    that play: what the stills draw."""
    return [source for source in scene["sources"] if source["name"] in names]


def video(args) -> None:
    """The run's audio (``<run>.wav``), one still per stop of the tour, and its video
    (``<run>.mp4``): the stills with a scrolling spectrogram in the panel at their foot
    (ffmpeg, unless ``--no-mp4``)."""
    from acoustic_spaces.demos.pictures import save, settings_lines

    pictures = sibling(__file__, "lakeside_pictures")
    folder = run_folder(args) / "seats"
    seats = pipeline.seats_of(folder, DEMO.command("render", args), args)
    tour = tour_of(args)
    keys = [f"{listener}.{weather}" for listener, weather, _, _ in tour]
    pipeline.missing_mixes(seats, keys, DEMO.command("render", args))
    gains = gains_of(seats, tour)
    stops = [
        pipeline.TourStop(key, start, start, stop, gains[key])
        for key, (_, _, start, stop) in zip(keys, tour, strict=True)
    ]
    audio = pipeline.tour_audio(
        stops,
        folder,
        seats,
        crossfade_s=CROSSFADE_S,
        fade_out_s=FADE_OUT_S,
        scene=args.scene,
    )
    sound = product(args, ".wav")
    write(sound, audio, seats["rate"])
    print(f"wrote {sound} ({len(audio) / seats['rate']:.1f} s)")
    scenes = {
        w: yaml.safe_load(s.scene.read_text(encoding="utf-8")) for w, s in weathers(args).items()
    }
    names = playing(args, list(CHANNELS))
    drawing = pictures.Pictures(
        {w: scene | {"sources": playing_sources(scene, names)} for w, scene in scenes.items()},
        seats["seats"],
        tour,
        TITLES,
        gains,
        pipeline.credited(played(args)["credit"], seats, names),
        folder=args.scene.parent,
        hrtf=args.hrtf,
        tool={
            w: settings_lines(
                DEMO, args, scenes[w], pipeline.read_json(s.results / pipeline.PARAMETERS)
            )
            for w, s in weathers(args).items()
        },
        playing=what_plays(names),
    )
    save(drawing.thumbnail(), product(args, "_thumbnail.png"))
    pipeline.video(
        run_folder(args),
        [(f"{listener}_{weather}", stop - start) for listener, weather, start, stop in tour],
        drawing.still,
        sound,
        out=product(args, ".mp4"),
        no_mp4=args.no_mp4,
        shown=lambda k: tour[k][0],
    )


DEMO = Demo(
    name="lakeside",
    example="lakeside_concert",
    title="A concert across a lake",
    description=__doc__,
    script=Path(__file__),
    steps=[
        Step(
            "music",
            music_step,
            "download the song and a signal per loudspeaker",
            music_done,
            music_download,
        ),
        Step(
            "scenes",
            scenes,
            "scene_demo.yaml, and scene_still.yaml in still air",
            lambda args: (
                pipeline.scene_made(args.scene, args.source)
                and pipeline.scene_made(args.still, args.source)
            ),
        ),
        SIMULATE,
        Step(
            "render", render, "every listener in both weathers, in --format (binaural by default)"
        ),
        Step("video", video, "the run's video, its audio and the stills, named after the run"),
    ],
    options=[
        Option(
            "song",
            "the song to play",
            "spicy_funk_cake",
            "choice",
            tuple(sorted(SONGS)),
            ("music",),
            named="plays",
            replaced_by="music",
        ),
        *music.own_song_options(),
        Option(
            "source",
            "the example's scene",
            Path("scene.yaml"),
            "file",
            steps=("scenes",),
            flag="--from",
            filter="Scene files (*.yaml *.yml)",
        ),
        Option("scene", "the scene the demo plays", Path("scene_demo.yaml"), "path"),
        Option("still", "the same in still air", Path("scene_still.yaml"), "path"),
        Option(
            "results",
            "simulate's folder for it, a folder in it for each output",
            Path("results_demo"),
            "folder",
        ),
        Option(
            "still_results",
            "simulate's folder for the scene in still air, a folder in it for each output",
            Path("results_still"),
            "folder",
        ),
        *common_options(),
        input_option(),
        *sources_options(
            help="only these loudspeakers, main_left, main_right or subs, or PA for all "
            "three, comma separated (default: all three): the demo mixes them alone and its "
            "stills draw them alone; the simulations keep every loudspeaker, so another "
            "pick needs no new ones",
            steps=("render", "video"),
        ),
        listeners_option(
            help="only these places, comma separated, in the order the video's tour visits "
            "them (default: every one, in the default tour's order): the demo mixes those "
            "alone, in both weathers, and the tour gives each its stops of the default tour "
            "and ends back at the first; the simulations keep every place, so another "
            "choice needs no new ones"
        ),
        Option(
            "tour",
            "listener:weather:phrases,... (weather evening or still; 8 bars a phrase), in "
            "place of the default tour; --listeners then leaves out the stops at the other "
            "places",
            None,
            steps=("video",),
        ),
        Option(
            "last_phrases",
            "phrases the last stop lasts (0: to the end of the song)",
            2.0,
            "float",
            steps=("video",),
        ),
    ],
    simulations=simulations,
    scene_from="scenes",
    plays=plays,
    problems=music.own_song_problems,
    listeners=listeners,
    sources=sources,
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
