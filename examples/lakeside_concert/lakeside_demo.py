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
    python lakeside_demo.py video     # demo/tour.wav, the stills, demo/tour.mp4

or `python lakeside_demo.py all`, which runs them in turn and leaves out what is there
already (the song, the scenes, the simulations). The scenes and their simulations go
beside the script (scene_demo.yaml and scene_still.yaml, results_demo and
results_still); the song, the feeds, the mixes and the video go to ./demo (--work). Use
headphones; `--hrtf me.sofa` plays through your own or measured HRTFs instead of a
spherical head (the simulations take it too). `--input subs=bass.wav` has a
loudspeaker (main_left, main_right or subs) play a file of yours instead of its part of
the song, at the level of that part. `--music song.wav --bpm 120` plays a song of your
own (WAV, or anything ffmpeg decodes) in place of --song: nothing is downloaded, the
tour moves every 8 bars of the tempo given, and the video credits the file by its name
unless `--credit` says what to show. The editor runs the same steps from Simulation >
Run demo…

From the audience to the far shore the sound drops by 24 to 25 dB(A), too much to
hear the far places in a video. So each stop of the tour is brought to the same loudness
(K-weighted, as loudness meters weigh it) as the audience's, and its caption says how
much quieter it really is. The window in still air keeps the gain of the window in the
evening's weather, so the difference the weather makes is heard as the simulation gives
it. Each place's mix is shifted earlier by the time the sound takes to reach it (1.6 s
across the lake), so that the beat carries on across the cuts. `--listeners
audience,bedroom_window` mixes those places alone, in both weathers, and the video tours
them alone, each stop of the tour there with its phrases.
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
    Step,
    common_options,
    input_option,
    listeners_option,
    music,
    picked,
    pipeline,
    sibling,
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
#: The scene's listener behind the stage, as far upwind as the park reaches, left out of
#: its own simulations.
UPWIND = "behind_stage"
#: How long the responses run: the far shore hears the stage 1.6 s late.
IR_LENGTH_S = 3.5

#: The default tour: listener, weather ("evening" or "still") and how many phrases it
#: lasts; the last one runs to the end of the song.
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


def song_path(args) -> Path:
    return music_folder(args) / f"{song_of(args)['name']}.wav"


def music_done(args) -> bool:
    """Whether the song and each loudspeaker's signal are there."""
    folder = music_folder(args)
    names = [song_of(args)["name"], *CHANNELS]
    if not all((folder / f"{name}.wav").exists() for name in names):
        return False
    # Signals levelled otherwise (the song's raw channels, before 4.1.0) are made again.
    song = folder / "song.json"
    return song.exists() and pipeline.read_json(song).get("levels") == FEED_LEVELS


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
    folder = music_folder(args)
    music.write_tracks({song["name"]: song["files"]}, folder, rate=RATE, channels=2)
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


def plays(args) -> dict[str, str]:
    """What each loudspeaker plays unless ``--input`` gives it a file: its channel of the
    song."""
    channels = {0: "left channel", 1: "right channel", "mono": "both channels in mono"}
    song = song_of(args)["name"]
    return {name: f"{song}, {channels[channel]}" for name, channel in CHANNELS.items()}


def played(args) -> dict:
    """The song ``music`` last wrote the loudspeakers' signals from."""
    path = music_folder(args) / "song.json"
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
        )
    print(f"wrote {args.scene} and {args.still}; now:")
    for simulation in simulations(args):
        print(f"  {pipeline.simulate_command(simulation, args.hrtf)}")


def weathers(args) -> dict[str, pipeline.Simulation]:
    """Each weather's scene and the folder its simulation writes."""
    return {
        "evening": pipeline.Simulation(args.scene, args.results, args.format),
        "still": pipeline.Simulation(args.still, args.still_results, args.format),
    }


def simulations(args) -> list[pipeline.Simulation]:
    """Both weathers, simulated one after the other."""
    return list(weathers(args).values())


def responses_of(args, weather: str) -> list[dict]:
    return pipeline.responses_of(weathers(args)[weather], args, DEMO.command("scenes", args))


# --------------------------------------------------------------------------- the mixes


def render(args) -> None:
    """Every listener's mix in both weathers (``--format``, binaural by default), with one
    gain for all: the loudest peak at -1 dBFS; and each one's level and delay."""
    feeds = {name: music_folder(args) / f"{name}.wav" for name in CHANNELS}
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
                    info={"weather": weather},
                )
            )
    pipeline.render_mixes(
        mixes, args.work / "seats", args, first=DEMO.command("scenes", args), inputs=args.inputs
    )


def tour_of(args) -> list[tuple[str, str, float, float]]:
    """(listener, weather, start, end) in the song's time; the stops end on the song's
    phrase boundaries (8 bars). The stops at the places ``--listeners`` leaves out are
    left out too, and stops at one place in one weather in a row are joined."""
    steps = TOUR
    if args.tour:
        steps = []
        for part in args.tour.split(","):
            fields = part.split(":")
            weather = fields[1] if len(fields) > 1 and fields[1] else "evening"
            steps.append((fields[0], weather, float(fields[2]) if len(fields) > 2 else 1.0))
    kept = set(picked(args, [listener for listener, _, _ in steps]))
    steps = [step for step in steps if step[0] in kept]
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
    """The places the scenes step puts in the demo's scenes, in the example's order: its
    listeners in use, and the one behind the stage, which the example leaves out of its
    own simulations."""
    data = yaml.safe_load(args.source.read_text(encoding="utf-8")) or {}
    return [
        item["name"]
        for item in data.get("listeners") or []
        if item.get("enabled", True) is not False or item["name"] == UPWIND
    ]


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


def video(args) -> None:
    """tour.wav, one still per stop of the tour, and tour.mp4: the stills with a
    scrolling spectrogram in the panel at their foot (ffmpeg, unless ``--no-mp4``)."""
    from acoustic_spaces.demos.pictures import save, settings_lines

    pictures = sibling(__file__, "lakeside_pictures")
    seats = pipeline.seats_of(args.work / "seats", DEMO.command("render", args))
    tour = tour_of(args)
    keys = [f"{listener}.{weather}" for listener, weather, _, _ in tour]
    pipeline.missing_mixes(seats, keys, DEMO.command("render", args))
    gains = gains_of(seats, tour)
    stops = [
        pipeline.TourStop(key, start, start, stop, gains[key])
        for key, (_, _, start, stop) in zip(keys, tour, strict=True)
    ]
    audio = pipeline.tour_audio(
        stops, args.work / "seats", seats, crossfade_s=CROSSFADE_S, fade_out_s=FADE_OUT_S
    )
    write(args.work / "tour.wav", audio, seats["rate"])
    print(f"wrote {args.work / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    scenes = {
        w: yaml.safe_load(s.scene.read_text(encoding="utf-8")) for w, s in weathers(args).items()
    }
    drawing = pictures.Pictures(
        scenes,
        seats["seats"],
        tour,
        TITLES,
        gains,
        pipeline.credited(played(args)["credit"], seats, list(CHANNELS)),
        folder=args.scene.parent,
        hrtf=args.hrtf,
        tool={
            w: settings_lines(
                DEMO, args, scenes[w], pipeline.read_json(s.results / pipeline.PARAMETERS)
            )
            for w, s in weathers(args).items()
        },
    )
    save(drawing.thumbnail(), args.work / "thumbnail.png")
    pipeline.video(
        args.work,
        [(f"{listener}_{weather}", stop - start) for listener, weather, start, stop in tour],
        drawing.still,
        args.work / "tour.wav",
        no_mp4=args.no_mp4,
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
            lambda args: args.scene.exists() and args.still.exists(),
        ),
        SIMULATE,
        Step("render", render, "every listener in both weathers, binaural"),
        Step("video", video, "tour.wav, the stills and tour.mp4 in the work folder"),
    ],
    options=[
        Option(
            "song",
            "the song to play",
            "spicy_funk_cake",
            "choice",
            tuple(sorted(SONGS)),
            ("music",),
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
        Option("results", "simulate's folder for it", Path("results_demo"), "folder"),
        Option(
            "still_results",
            "simulate's folder for the scene in still air",
            Path("results_still"),
            "folder",
        ),
        *common_options(),
        input_option(),
        listeners_option(
            help="only these places, comma separated (default: every one): the demo mixes "
            "those alone, in both weathers, and its video's tour leaves the others out; the "
            "simulations keep every place, so another choice needs no new ones"
        ),
        Option(
            "tour",
            "listener:weather:phrases,... (weather evening or still; 8 bars a phrase)",
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
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
