"""A song played on the lakeside concert example's stage, heard in the audience, behind
the stage, from a boat and in the village across the lake, and a video that tours them.

The stage's sound system, two line arrays and subwoofers, plays a stereo recording
(--song): by default
"Spicy Funk Cake" by the Rhythmussportgruppe, a nine-piece funk band, recorded for the
German Acoustical Society (DEGA) at Hochschule Düsseldorf and published under CC BY 4.0
(https://doi.org/10.5281/zenodo.3601032). At a bedroom window across the
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
the song, at the level of that part. The editor runs the same steps from Simulation >
Run demo…

From the audience to the far shore the sound drops by 20 to 25 dB(A), too much to
hear the far places in a video. So each stop of the tour is brought to the same loudness
(K-weighted, as loudness meters weigh it) as the audience's, and its caption says how
much quieter it really is. The window in still air keeps the gain of the window in the
evening's weather, so the difference the weather makes is heard as the simulation gives
it. Each place's mix is shifted earlier by the time the sound takes to reach it (1.6 s
across the lake), so that the beat carries on across the cuts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from acoustic_spaces.demos import (
    SIMULATE,
    Demo,
    Option,
    Step,
    common_options,
    input_option,
    music,
    pipeline,
    sibling,
)
from acoustic_spaces.demos.audio import mono, read, write
from acoustic_spaces.demos.pictures import save, settings_lines

#: The songs the demo can play (``--song``): the files each one is made of (one here,
#: a member of a zip archive read by HTTP range requests or a file of its own, as
#: :mod:`acoustic_spaces.demos.music` fetches them), its tempo (beats a minute; the tour
#: moves on every 8 bars) and the credit its licence asks for.
SONGS = {
    "spicy_funk_cake": {
        "files": [
            (
                "https://zenodo.org/records/3601032/files/Rhythmussportgruppe_Preview.zip",
                "Rhythmussportgruppe_Preview/Spicy_Fun_cake_Preview/1_REFERENCE-MIX/"
                "Spicy_Funk_Cake.wav",
            )
        ],
        "bpm": 110.0,
        "credit": 'Music: Rhythmussportgruppe, "Spicy Funk Cake" (Niklas Dahlheimer), recorded '
        "for the DEGA at Hochschule Düsseldorf, CC BY 4.0 · doi.org/10.5281/zenodo.3601032",
    },
    "great_british_summer": {
        "files": [
            (
                "https://zenodo.org/records/19683001/files/masters_stereo.zip",
                "masters_stereo/09 - Great British Summer.wav",
            )
        ],
        "bpm": 120.0,
        "credit": 'Music: Tom McKenzie, "Great British Summer" (album "Extraordinary", AlbumDB), '
        "CC BY 4.0 · doi.org/10.5281/zenodo.19683001",
    },
    "potassium": {
        "files": [
            (
                "https://raw.githubusercontent.com/ichbinsoftware/everythingisfree/"
                "e1d0d5c0422d2626c211d5d877e6f69c37ce8712/src/4.Potassium/4.Potassium_Master.m4a",
                None,
            )
        ],
        "bpm": 90.0,
        "credit": 'Music: Software-Entwicklungskit, "Potassium" (album "Everything is Free"), '
        "CC0 1.0 · github.com/ichbinsoftware/everythingisfree",
    },
}
#: The rate the songs are brought to, that of the simulation's responses.
RATE = 48000
#: What ``music`` keeps of a song in song.json.
SONG_KEPT = ("name", "bpm", "credit")


#: The stage's sound system, as a lawn concert hangs it: two line arrays of eight boxes
#: (lake_array.yaml, the box's own pattern in box.csv, both made with the stadium
#: example's make_model.py) flown 6.5 m up on the audience's left (-y: the audience looks
#: towards -x) and right, aimed straight down the lawn and, box by box, from 5 degrees
#: down at the last rows, where the lawn meets the water, to 36 at the front rows; and
#: a block of four cardioid subwoofers
#: on the ground under the middle of the stage front, 2 m nearer the audience than the
#: arrays and delayed by as much, so that both arrive together there. Levels are sound
#: power (dB re 1 pW); the arrays play the song's channels, the subwoofers both. The
#: system is tuned at the audience, as a system engineer tunes it (eq_db per octave band,
#: 63 Hz to 8 kHz): the subwoofers rolled off above 125 Hz and 3 dB over the arrays
#: together, the arrays' low mids up and their top octave up by what 30 m of air takes,
#: so that at the audience every octave from 250 Hz to 4 kHz is within 1 dB of 1 kHz,
#: 8 kHz 1.5 dB down and the two lowest 2 dB up.
SYSTEM = {
    "main_left": {
        "position": [0.0, -6.0, 6.5],
        "orientation": {"yaw": 0.0},
        "level_db": 124.0,
        "eq_db": [0, 0, 5, 0, 0, 1, 1, 2],
        "directivity": {"file": "lake_array.yaml", "near_field": True},
        "channel": 0,
    },
    "main_right": {
        "position": [0.0, 6.0, 6.5],
        "orientation": {"yaw": 0.0},
        "level_db": 124.0,
        "eq_db": [0, 0, 5, 0, 0, 1, 1, 2],
        "directivity": {"file": "lake_array.yaml", "near_field": True},
        "channel": 1,
    },
    "subs": {
        "position": [1.0, 0.0, 0.6],
        "level_db": 130.0,
        "eq_db": [0, 0, -18, -40, -40, -40, -40, -40],
        "delay_ms": 6,
        "directivity": {"file": "sub_array.yaml", "near_field": True},
        "channel": "mono",
    },
}
#: A listener behind the stage, as far upwind as the park reaches.
UPWIND = {"name": "behind_stage", "position": [-140.0, 0.0, 1.5], "orientation": {"yaw": 0.0}}
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
    return SONGS[args.song] | {"name": args.song}


def phrase_s(song: dict) -> float:
    """8 bars of ``song``, s."""
    return music.phrase_s(song)


def music_folder(args) -> Path:
    return args.work / "music"


def song_path(args) -> Path:
    return music_folder(args) / f"{args.song}.wav"


def music_done(args) -> bool:
    """Whether the song and each loudspeaker's signal are there."""
    folder = music_folder(args)
    names = [args.song, *SYSTEM]
    return all((folder / f"{name}.wav").exists() for name in names)


def music_step(args) -> None:
    """Download ``--song`` (one member of a zip archive, or the file itself), bring it to
    48 kHz stereo, and write what each loudspeaker plays (SYSTEM, "channel"). A song
    already in the folder is kept."""
    song = song_of(args)
    folder = music_folder(args)
    music.write_tracks({args.song: song["files"]}, folder, rate=RATE, channels=2)
    x, rate = read(song_path(args))
    for name, source in SYSTEM.items():
        channel = source["channel"]
        write(folder / f"{name}.wav", mono(x) if channel == "mono" else x[:, channel], rate)
    pipeline.write_json(folder / "song.json", {k: v for k, v in song.items() if k in SONG_KEPT})
    print(f"wrote {', '.join(SYSTEM)} ({len(x) / rate:.0f} s of {args.song}) to {folder}")


def plays(args) -> dict[str, str]:
    """What each loudspeaker plays unless ``--input`` gives it a file: its channel of the
    song."""
    channels = {0: "left channel", 1: "right channel", "mono": "both channels in mono"}
    return {name: f"{args.song}, {channels[s['channel']]}" for name, s in SYSTEM.items()}


def played(args) -> dict:
    """The song ``music`` last wrote the loudspeakers' signals from."""
    path = music_folder(args) / "song.json"
    if not path.exists():
        raise SystemExit(f"error: no song yet; run first: {DEMO.command('music', args)}")
    return pipeline.read_json(path)


# --------------------------------------------------------------------------- the scenes


def scenes(args) -> None:
    """``--scene`` (scene_demo.yaml) from ``--from`` (scene.yaml), with the stage's
    sound system (SYSTEM) in place of the example's one source, a listener behind the
    stage and responses long enough for the far shore; and ``--still``
    (scene_still.yaml), the same in still air."""
    scene = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    scene["name"] = scene["name"] + " (demo)"
    scene["sources"] = [
        {"name": name} | {k: v for k, v in source.items() if k != "channel"}
        for name, source in SYSTEM.items()
    ]
    if UPWIND["name"] not in {item["name"] for item in scene["listeners"]}:
        scene["listeners"].append(dict(UPWIND))
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
    feeds = {name: music_folder(args) / f"{name}.wav" for name in SYSTEM}
    mixes = []
    for weather, simulation in weathers(args).items():
        for listener in sorted({e["listener"] for e in responses_of(args, weather)}):
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
    phrase boundaries (8 bars)."""
    steps = TOUR
    if args.tour:
        steps = []
        for part in args.tour.split(","):
            fields = part.split(":")
            weather = fields[1] if len(fields) > 1 and fields[1] else "evening"
            steps.append((fields[0], weather, float(fields[2]) if len(fields) > 2 else 1.0))
    song = played(args)
    x, rate = read(song_path(args))
    tour = music.phrase_tour(
        [((listener, weather), phrases) for listener, weather, phrases in steps],
        len(x) / rate,
        phrase_s(song),
        last_phrases=args.last_phrases,
    )
    return [(listener, weather, start, stop) for (listener, weather), start, stop in tour]


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
        pipeline.credited(played(args)["credit"], seats, list(SYSTEM)),
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
        Step("music", music_step, "download the song and a signal per loudspeaker", music_done),
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
)

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
