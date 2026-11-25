"""A song played by a band on the stadium example's stage and through its sound system,
heard at each of its listeners, and a video that tours them.

The song (--song) is "Spicy Funk Cake" by the Rhythmussportgruppe (composed by Niklas
Dahlheimer), recorded for the sound database of the DEGA (German Acoustical Society) at
Hochschule Düsseldorf, CC BY 4.0, https://doi.org/10.5281/zenodo.3601032, or "Francium"
or "Rubidium" by Software-Entwicklungskit from the album "Everything is Free", CC0 1.0,
https://github.com/ichbinsoftware/everythingisfree, or a song of your own (--music, its
stereo master, with --bpm its tempo and --stem its stems). The sound system (main arrays,
subwoofers, fills, towers and delays) plays the song's stereo master; the monitors play
mixes of its stems: the singer's wedges the vocal, the drum fill the drums and bass, the
side fills a mix for the whole stage; and the band on the stage plays its own parts: the
drum kit the drums, the singer the vocal, the guitarist's amplifier the guitar (the synth
lead, for the album's songs). Run it from the folder of the stadium example
(`acoustic-spaces examples --copy . stadium_concert && cd stadium_concert`), in the
Python environment acoustic-spaces is installed in, with ffmpeg on the PATH:

    python stadium_demo.py music     # download the master and the stems (42 to 313 MB)
    python stadium_demo.py band      # scene_band.yaml: the scene with the band on the stage
    python stadium_demo.py simulate  # simulate scene_band.yaml into results_band, binaural
    python stadium_demo.py feeds     # one signal per loudspeaker, as loud as the scene says
    python stadium_demo.py render    # every listener, binaural
    python stadium_demo.py video     # demo/stadium_spicy_funk_cake_binaural.mp4

or `python stadium_demo.py all`, which runs them in turn and leaves out what is there
already (the music, the scene, the simulation, the feeds, the mixes). The scene and its
simulation go beside the script (scene_band.yaml, and results_band/binaural for the
binaural output: each output in a folder of its own); the music and each song's feeds go
to ./demo (--work), and each run's mixes and stills to a folder there named after its
settings (what it plays, the listeners, the output: demo/stadium_francium_binaural/),
with the video, its audio and its thumbnail beside it, so that a run of another song or
with other settings writes over none of them.
Use headphones; `--hrtf me.sofa` plays through your own or measured HRTFs instead of a
spherical head (the simulation takes it too). `--input drum_kit=drums.wav` has a source
play a file of yours instead of its feed, at the level of that feed (a name the scene
has no source for is refused with the list of them). The editor runs the same steps
from Simulation > Run demo…

A song of your own: `--music song.wav --bpm 120` (WAV, or anything ffmpeg decodes)
plays its master through the sound system in place of --song's; `--stem vocal=vox.wav
--stem drums=drums.wav` gives the stems the monitors and the band play (drums, bass,
guitar, keys, brass, vocal, backing; a mix none of whose stems is given plays nothing,
so without stems the monitors and the band are silent); `--downbeat 0.5` says when the
first downbeat falls, so that the tour's cuts stay on the beat; `--credit` is the line
the video shows for it (the file's name unless given). Nothing is downloaded; the music
goes to demo/music/<file name>, so each song keeps its own folder.

The impulse responses `simulate` writes carry each source's feed (gain_db, eq_db,
delay_ms) but are per unit of sound power. `feeds` scales each loudspeaker's signal by
its level_db against the loudest one's, after bringing every signal to the same gated
RMS, so the balance between the main arrays (134 dB), the fills (116 to 130 dB) and
the monitors is the scene's, and the feed files are what each loudspeaker plays.
`render` runs `auralize` with `--equal-power`, so the levels are applied once.

`video` plays the song once, moving between listeners on phrase boundaries (every 8
bars), with a 1 s crossfade. Each listener's mix is shifted
earlier by the time the sound system takes to reach them (0.1 s at the mix position,
0.4 s at the far end), so that the beat carries on across the cut; the caption gives
that time and the level there against the loudest listener's. On stage it is 6 to 18
dB(A) louder than in the audience, too much to hear both in one video, so each stop is
turned up to the loudness of the loudest one (K-weighted, as loudness meters weigh it)
and its caption says by how much; the mixes of a run keep one gain for all.
`--listeners east_stand,mix_position` mixes those listeners alone, and the video tours
them in that order, each stop with its phrases and the last to the end of the song.
`--sources PA` mixes the sound system alone (or `Monitors`, `Band`, a group such as
`PA/Delays` or a source by its name, comma separated), each at the level the scene
gives it against all the others, from the same simulation and feeds;
`--sources-play in-turn` plays the sound system, the monitors and the band one after
another, each alone, in one video.
"""

from __future__ import annotations

import functools
import math
import sys
import urllib.parse
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import yaml

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
    music,
    name_part,
    picked,
    pipeline,
    playing,
    product,
    run_folder,
    sibling,
    source_matches,
    sources_options,
)
from acoustic_spaces.demos.audio import (
    channels_rms,
    gated_rms,
    lag_of,
    read,
    resampled,
    shifted,
    write,
)

#: The songs the demo can play (``--song``), each from its release, only the files it
#: needs: the master and the stems by the names the mixes use, each a list of files
#: summed, a file being ``(url, member)``, a member of the zip archive at the URL read
#: by HTTP range requests, or ``(url, None)``, the file at the URL (M4A files are decoded
#: with ffmpeg); the tempo (beats a minute); the first downbeat (s), which the 8 bar
#: phrases the tour moves on run from; the tour's changes to the default (listener:
#: phrases); the credit its licence asks for; and the text of SOURCES.txt.
RSG = "https://zenodo.org/records/3601032/files/RSG_Complete_Project.zip"
RSG_SONG = "RSG_Complete_Project/Abgabe_RSG_DEGA/1_Audio/2_SPICY_FUNK_CAKE/"
RSG_STEMS = RSG_SONG + "2_STEMS_PROCESSED/"
ALBUM = "https://github.com/ichbinsoftware/everythingisfree"
ALBUM_COMMIT = "e1d0d5c0422d2626c211d5d877e6f69c37ce8712"
ALBUM_RAW = f"https://raw.githubusercontent.com/ichbinsoftware/everythingisfree/{ALBUM_COMMIT}/src/"
#: "Francium" starts a beat into a bar, and its first phrase, the intro, is a bar short:
#: its phrases run from 14.76 s, and the tour's first stop takes the intro with the first.
FRANCIUM_BPM = 127.75
FRANCIUM_DOWNBEAT_S = 14.76


def album_song(number: int, title: str, groups: dict[str, list[str]], **more) -> dict:
    """A song of "Everything is Free" (Software-Entwicklungskit, CC0): its master and its
    stems, in the groups the mixes use. The album is electronic: the amplifier on the
    stage plays its synth lead, the drum kit the beats."""
    folder = f"{ALBUM_RAW}{number}.{title}/"
    return {
        "master": [(folder + urllib.parse.quote(f"{number}.{title}_Master.m4a"), None)],
        "stems": {
            name: [
                (folder + urllib.parse.quote(f"{number}.{title}_Stem_{stem}.m4a"), None)
                for stem in stems
            ]
            for name, stems in groups.items()
        },
        "credit": f'Music: Software-Entwicklungskit, "{title}" (album "Everything is Free"), '
        "CC0 1.0 · github.com/ichbinsoftware/everythingisfree",
        "sources": f'Software-Entwicklungskit, "{title}" from the album "Everything is Free",\n'
        f"{ALBUM} (commit {ALBUM_COMMIT}), src/{number}.{title}: the master and the stems\n"
        "License: CC0 1.0, https://creativecommons.org/publicdomain/zero/1.0/\n",
    } | more


#: The songs the demo can play (``--song``): the master and the stems, the MB they take
#: to download (``music`` reads only them out of an archive), the tempo and the first
#: downbeat, the tour's changes, the credit and what SOURCES.txt says of them.
SONGS = {
    "spicy_funk_cake": {
        "master": [(RSG, RSG_SONG + "1_REFERENCE-MIX/Spicy_Funk_Cake.wav")],
        # The shouts are the band's backing vocals.
        "stems": {
            "drums": [(RSG, RSG_STEMS + "1_STEM_PROCESSED_DRUMS.wav")],
            "bass": [(RSG, RSG_STEMS + "2_STEM_PROCESSED_Bass.wav")],
            "guitar": [(RSG, RSG_STEMS + "3_STEM_PROCESSED_GIT.wav")],
            "keys": [(RSG, RSG_STEMS + "4_STEM_PROCESSED_KEYS.wav")],
            "brass": [(RSG, RSG_STEMS + "5_STEM_PROCESSED_BRASS.wav")],
            "vocal": [(RSG, RSG_STEMS + "6_STEM_PROCESSED_VOX.wav")],
            "backing": [(RSG, RSG_STEMS + "7_STEM_PROCESSED_SHOUTS.wav")],
        },
        "download_mb": 313,
        "bpm": 110.0,
        # After a pickup; the song is eleven phrases long, so the tour leaves out the
        # side of the pitch and the north stand's near seats.
        "downbeat_s": 0.5,
        "tour": {"pitch_side": 0, "north_stand_near": 0},
        "credit": 'Music: Rhythmussportgruppe, "Spicy Funk Cake" (Niklas Dahlheimer), recorded '
        "for the DEGA at Hochschule Düsseldorf, CC BY 4.0 · doi.org/10.5281/zenodo.3601032",
        "sources": 'Rhythmussportgruppe, "Spicy Funk Cake" (composed by Niklas Dahlheimer),\n'
        "recorded for the DEGA sound database at Hochschule Düsseldorf (2017),\n"
        "https://doi.org/10.5281/zenodo.3601032, RSG_Complete_Project.zip:\n"
        f"{RSG_SONG}1_REFERENCE-MIX/Spicy_Funk_Cake.wav and the stems in {RSG_STEMS}\n"
        "License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n",
    },
    "francium": album_song(
        7,
        "Francium",
        {
            "drums": [
                "BIG BEAT",
                "GARAGE BEAT",
                "FILL BEAT",
                "SCRATCH BEAT",
                "BREAK",
                "KICK",
                "HATS",
                "OPEN HATS",
                "CYMBALS",
                "CRASH",
                "CLAPS",
                "PERC BUILD",
                "FILL BUILD",
                "FILL SCRATCH",
            ],
            "bass": ["SINE"],
            "guitar": ["MINILOGUE_SYNTH"],
            "keys": ["MELODY GLITCH", "MICROCOSM_EFFECT", "TEXTURE"],
            "vocal": ["VOX LEAD"],
            "backing": [
                "BGVOX ALT OUTRO",
                "BGVOX BOP",
                "BGVOX BUILD",
                "BGVOX ECHO",
                "BGVOX OUTRO",
                "BGVOX SCATTER OUTRO",
            ],
        },
        download_mb=42,
        bpm=FRANCIUM_BPM,
        downbeat_s=FRANCIUM_DOWNBEAT_S,
        # Nineteen phrases and a half: the stops that take a section's start last two.
        tour={
            "pitch_front": 2,
            "north_stand_top": 2,
            "east_stand": 2,
            "pitch_back": 2,
            "pitch_behind_towers": 2,
        },
    ),  # fmt: skip
    "rubidium": album_song(
        5,
        "Rubidium",
        {
            "drums": ["DRUMS MAIN", "DRUMS 2", "DRUMS OUTRO", "BUILD"],
            "guitar": ["MINILOGUE_SYNTH"],
            "keys": ["PLUCK", "MELODY BREAK"],
            "vocal": ["VOX LEAD"],
            "backing": ["BGVOX"],
        },
        download_mb=43,
        bpm=132.0,
        downbeat_s=0.0,  # the master starts on its first downbeat
        tour={},
    ),
}
#: What ``music`` writes of a song to song.json and ``feeds`` to feeds.json.
SONG_KEPT = ("name", "bpm", "downbeat_s", "tour", "credit")
#: The stems a song of your own may give (``--stem``): those the mixes below name.
STEMS = ("drums", "bass", "guitar", "keys", "brass", "vocal", "backing")

#: Monitor mixes: stem and its gain in dB (stems a song does not have are left out).
SINGER_MIX = {"vocal": 0, "backing": -6, "keys": -8, "guitar": -12, "bass": -14, "drums": -16}
DRUMMER_MIX = {"drums": 0, "bass": 0, "vocal": -6, "guitar": -8, "keys": -8}
# The side fills carry the song for the whole stage, with the drums well down: the kit
# itself is loud on stage, so the players hear the drums from where the drummer is.
STAGE_MIX = {"vocal": 3, "drums": -20}  # every other stem at 0 dB
MONITORS = {"vocal_wedges": "singer", "drum_fill": "drummer", "side_fill": "stage"}
#: Each monitor mix's stems and gains; None: every stem the song has, at 0 dB unless named.
MONITOR_MIXES: dict[str, tuple[dict[str, float], float | None]] = {
    "singer": (SINGER_MIX, None),
    "drummer": (DRUMMER_MIX, None),
    "stage": (STAGE_MIX, 0.0),
}
#: The band itself: the scene's sources in this group, left out of its simulations until
#: ``band`` takes them in, and the stem each plays.
BAND = "Band"
BAND_MIX = {"drum_kit": {"drums": 0}, "singer_voice": {"vocal": 0}, "guitar_amp": {"guitar": 0}}

#: The default tour: listener and how many phrases it lasts (0.5: 4 bars); the last one
#: runs to the end. Each song changes it as it fits (``SONGS``, "tour"; 0 leaves a stop
#: out).
TOUR = [
    ("mix_position", 1),
    ("pitch_front", 1),
    ("singer", 1),
    ("drummer", 1),
    ("guitarist", 1),
    ("pitch_side", 1),
    ("north_stand_near", 1),
    ("north_stand_far", 1),
    ("north_stand_top", 1),
    ("east_stand", 1),
    ("pitch_back", 1),
    ("pitch_behind_towers", 1),
    ("mix_position", 0),
]
TITLES = {
    "mix_position": ("Mix position", "on the pitch, where the sound engineer stands"),
    "pitch_front": ("Front of the pitch", "close to the stage, between the main arrays"),
    "pitch_side": ("Side of the pitch", "off-axis, towards the south stand"),
    "pitch_behind_towers": ("Behind the delay towers", "on the pitch, 18 m past the towers"),
    "pitch_back": ("Back of the pitch", "the far end, in front of the east stand"),
    "north_stand_near": ("North stand, near", "low in the side stand, close to the stage"),
    "north_stand_far": ("North stand, far", "in the side stand, under a delay"),
    "north_stand_top": ("North stand, top row", "the last row, 24 m up, under the roof"),
    "east_stand": ("East stand", "opposite the stage, the farthest seats"),
    "singer": ("On stage: the singer", "between the vocal wedges"),
    "drummer": ("On stage: the drummer", "beside the drum fill"),
    "guitarist": ("On stage: the guitarist", "in the side fills' sound"),
}
#: What the stills call the concert's groups of sources (the scene's ``group``), and
#: shorter, for the thumbnail.
GROUP_WORDS = {
    "PA": "sound system",
    "PA/Mains": "main arrays",
    "PA/Subwoofers": "subwoofers",
    "PA/Fills": "fills",
    "PA/Delays": "delays",
    "Monitors": "stage monitors",
    "Band": "band",
}
SHORT_WORDS = {"Monitors": "monitors"}
CROSSFADE_S = 1.0
FADE_OUT_S = 4.0
#: What ``--sources`` and ``--listeners`` say of the stadium demo.
SOURCES_HELP = (
    "only these sources, by name or by the scene's group, comma separated: PA (its groups "
    "PA/Mains, PA/Subwoofers, PA/Fills and PA/Delays), Monitors, Band (default: every "
    "source of the concert; the helicopter is helicopter_demo.py's); the simulation and "
    "the feeds keep every source, so another pick needs neither again, and each source "
    "plays as loud against the others as the scene has it"
)
LISTENERS_HELP = (
    "only these listeners, in this order, comma separated (default: every one, the tour "
    "in the demo's order): the demo mixes those alone and its video tours them in this "
    "order, each for its phrases and the last to the end of the song; the simulation "
    "keeps every listener, so another choice needs no new one"
)


# --------------------------------------------------------------------------- the music


def song_of(args) -> dict:
    """The song: ``--song``'s preset, or the song of your own ``--music`` gives, named
    after its file, with the stems of ``--stem``, the tempo of ``--bpm``, the first
    downbeat of ``--downbeat``, the default tour and the credit of ``--credit`` or the
    file's name."""
    if args.music is None:
        return SONGS[args.song] | {"name": args.song}
    stems = {name: Path(file) for name, file in (args.stems or {}).items()}
    return {
        "name": args.music.stem,
        "master": args.music,
        "stems": stems,
        "bpm": args.bpm,
        "downbeat_s": args.downbeat,
        "tour": {},
        "credit": args.credit or f"Music: {args.music.name}",
        "sources": f"master: {args.music}\n"
        + "".join(f"{name}: {file}\n" for name, file in stems.items()),
    }


def stems_of(what: str, stems: set[str]) -> list[str]:
    """The stems of ``stems`` (the song's) that the feed ``what`` plays: a monitor mix's
    or a player's, in the order the song has them."""
    if what in BAND_MIX:
        plays = set(BAND_MIX[what])
    elif what in MONITOR_MIXES:
        gains, others = MONITOR_MIXES[what]
        plays = stems if others is not None else set(gains)
    else:
        return []
    return [name for name in STEMS if name in stems and name in plays]


def phrase_s(song: dict) -> float:
    """A phrase of ``song`` (8 bars), s. The tour moves on phrase boundaries: a crossfade
    that starts half a second before one ends on its downbeat."""
    return music.phrase_s(song)


def music_folder(args) -> Path:
    return args.work / "music" / song_of(args)["name"]


def music_done(args) -> bool:
    """Whether the song's master and stems are in the folder; for a song of your own,
    made from the files given as they are now, with its tempo, downbeat and credit."""
    folder = music_folder(args)
    song = song_of(args)
    wavs = {name: folder / f"{name}.wav" for name in ("master", *song["stems"])}
    if not all(path.exists() for path in wavs.values()):
        return False
    if args.music is None:
        return True
    given = {"master": song["master"]} | song["stems"]
    if any(
        Path(file).exists() and wavs[name].stat().st_mtime < Path(file).stat().st_mtime
        for name, file in given.items()
    ):
        return False
    path = folder / "song.json"
    kept = {k: v for k, v in song.items() if k in SONG_KEPT}
    return path.exists() and pipeline.read_json(path) == kept


def music_download(args) -> str | None:
    """What ``music`` downloads: ``--song``'s master and stems, nothing for a song of
    your own."""
    if args.music is not None:
        return None
    song = SONGS[args.song]
    stems = sum(len(files) for files in song["stems"].values())
    return f"the master and the {stems} stems of {args.song} ({song['download_mb']} MB)"


def music_step(args) -> None:
    """Download ``--song``'s master and stems: only the files the song names, out of its
    release's archives or one by one, each stem's files decoded and summed. Files
    already in the folder are kept. For a song of your own (``--music``, ``--stem``),
    the files are decoded and written to the folder, nothing downloaded."""
    song = song_of(args)
    folder = music_folder(args)
    music.write_tracks({"master": song["master"]} | song["stems"], folder)
    (folder / "SOURCES.txt").write_text(song["sources"], encoding="utf-8")
    pipeline.write_json(folder / "song.json", {k: v for k, v in song.items() if k in SONG_KEPT})


# --------------------------------------------------------------------------- the scene


def band(args) -> None:
    """``--scene`` (default scene_band.yaml) written from ``--from`` (scene.yaml) with the
    band on the stage, its Band group taken into the simulation: the drum kit, the
    singer's voice and the guitarist's amplifier, so that the players are heard from
    where they stand and not only through the loudspeakers."""
    scene = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    scene["sources"] = with_band(scene["sources"], args.source)
    pipeline.write_scene(
        args.scene,
        scene,
        "# The stadium concert example with the band on the stage, written by stadium_demo.py.\n",
        args.source.parent,
        made_from=args.source,
    )
    print(f"wrote {args.scene}; now: {pipeline.simulate_command(simulations(args)[0], args.hrtf)}")


def scene_of(args) -> dict:
    return yaml.safe_load(args.scene.read_text(encoding="utf-8"))


def with_band(sources: list[dict], path: Path) -> list[dict]:
    """The scene's sources (``path``'s) with its band in use."""
    if not any(s.get("group") == BAND for s in sources):
        raise SystemExit(f"error: {path} has no band: no source in the {BAND} group")
    return [
        {k: v for k, v in s.items() if k != "enabled"} if s.get("group") == BAND else s
        for s in sources
    ]


def in_use(sources: list[dict]) -> list[dict]:
    """The sources the concert's simulations take: not those left out of them
    (``enabled: false``), as the helicopter is until helicopter_demo.py flies it."""
    return [s for s in sources if s.get("enabled", True) is not False]


def concert_file(args) -> tuple[Path, bool] | None:
    """The scene the concert's sources are read from, and whether its band is to be
    taken in: the scene with the band (``--scene``), or before ``band`` writes it,
    ``--from``; None without either."""
    if args.scene.exists():
        return args.scene, False
    if args.source.exists():
        return args.source, True
    return None


def every_source(args) -> list[dict]:
    """Every source of the concert's scene (:func:`concert_file`), the band in use, those
    left out of its simulations too (the helicopter); none without a scene."""
    found = concert_file(args)
    if found is None:
        return []
    path, band = found
    sources = yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]
    return with_band(sources, path) if band else sources


def sources(args) -> list[Source]:
    """The sources a run can play (``--sources``): those the concert's simulation takes
    (:func:`in_use`), each with its group in the scene (``PA/Mains``, ``Monitors``,
    ``Band``), all on. The simulation and the feeds keep every one, so a pick needs
    neither again, and each plays at the level the scene gives it against the loudest
    of them all."""
    found = concert_file(args)
    if found is None:
        return []
    path, band = found
    stat = path.stat()
    named = _sources_in(str(path.resolve()), band, stat.st_mtime_ns, stat.st_size)
    return [Source(name, group) for name, group in named]


@functools.lru_cache(maxsize=8)
def _sources_in(path: str, band: bool, _mtime_ns: int, _size: int) -> tuple:
    sources = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["sources"]
    if band:
        sources = with_band(sources, Path(path))
    return tuple((s["name"], s.get("group")) for s in in_use(sources))


def silent(what: str, stems: Collection[str]) -> bool:
    """Whether a feed of the kind ``what`` (:func:`role`) plays nothing of a song with
    ``stems``: a monitor mix or a player none of whose stems it has (a song of your own
    given without them)."""
    return (what in BAND_MIX or what in MONITOR_MIXES) and not stems_of(what, set(stems))


def sources_problem(args) -> str | None:
    """What is wrong with the sources ``--sources`` picks (``args`` resolved): one left
    out of the concert's simulations (the helicopter, which helicopter_demo.py flies
    over it), or a pick of none but silent feeds (the monitors and the band of a song
    of your own given without its stems)."""
    names = given_names(getattr(args, "sources", None))
    if not names:
        return None
    try:
        every = every_source(args)
    except SystemExit as exc:  # a scene with no band
        return str(exc).removeprefix("error: ")
    used = in_use(every)
    known = [Source(s["name"], s.get("group")) for s in used]
    left_out = [Source(s["name"], s.get("group")) for s in every if s not in used]
    for name in names:
        if any(source_matches(source, name) for source in known):
            continue
        hit = [source.name for source in left_out if source_matches(source, name)]
        if hit:
            fly = "; helicopter_demo.py flies it over the concert" if "helicopter" in hit else ""
            return (
                f"--sources {name}: left out of the concert's simulation, so it plays "
                f"nothing here{fly}"
            )
    chosen = getattr(args, "playing", None)
    if not chosen:
        return None
    song = song_of(args)
    if all(silent(role(s), song["stems"]) for s in used if s["name"] in chosen):
        return (
            f"--sources {','.join(names)}: every source it picks is silent, as "
            f"{song['name']} has no stem for them: give the stems (--stem), or pick the "
            "sound system (PA) too"
        )
    return None


def problems(demo: Demo, args) -> str | None:
    """A song of your own given as it should be (:func:`music.own_song_problems`), and
    sources to play that the concert has and that play something
    (:func:`sources_problem`)."""
    return music.own_song_problems(demo, args) or sources_problem(demo.resolved(args))


def simulations(args) -> list[pipeline.Simulation]:
    """The one simulation the demo mixes from: the scene with the band, the mixes' output,
    into that output's folder in ``--results``
    (:func:`~acoustic_spaces.demos.pipeline.results_for`)."""
    return [pipeline.Simulation(args.scene, pipeline.results_for(args.results, args), args.format)]


def responses_of(args) -> list[dict]:
    return pipeline.responses_of(simulations(args)[0], args, DEMO.command("band", args))


# --------------------------------------------------------------------------- the feeds


def total_level(level: float | list[float]) -> float:
    if isinstance(level, list):
        return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in level))
    return float(level)


def role(source: dict) -> str:
    """What a source plays: a monitor mix, or the channel of the master its ``channel``
    names in the scene; a loudspeaker of a scene that names none plays the master in
    mono (subwoofers and loudspeakers on the stage's axis) or its left or right
    channel, as the audience sees it (facing the stage, which is at -x: left is -y)."""
    if source["name"] in BAND_MIX:
        return source["name"]
    for prefix, mix in MONITORS.items():
        if source["name"].startswith(prefix):
            return mix
    if source.get("channel") is not None:
        return source["channel"]
    y = source["position"][1]
    if "sub" in source["name"] or abs(y) < 1.0:
        return "mono"
    return "left" if y < 0 else "right"


#: What each kind of feed (:func:`role`) is, for the editor's map of inputs.
ROLES = {
    "left": "the master's left channel, on the audience's left",
    "right": "the master's right channel, on the audience's right",
    "mono": "the master in mono",
    "singer": "the singer's monitor mix",
    "drummer": "the drummer's monitor mix",
    "stage": "the stage's mix, drums well down",
}


def plays(args) -> dict[str, str]:
    """What each source that plays (``--sources``) plays unless ``--input`` gives it a
    file (:func:`role`): the sources of the scene with the band, or, before ``band``
    writes it, of ``--from`` and the band."""
    used = in_use(every_source(args))
    chosen = set(playing(args, [source["name"] for source in used]))
    song = song_of(args)
    stems = set(song["stems"])
    out = {}
    for source in used:
        if source["name"] not in chosen:
            continue
        what = role(source)
        if what in BAND_MIX:
            have = stems_of(what, stems)
            out[source["name"]] = (
                f"{song['name']}, the {', '.join(have)} stem"
                if have
                else f"nothing: {song['name']} has no {', '.join(BAND_MIX[what])} stem"
            )
        elif silent(what, stems):
            out[source["name"]] = f"nothing: {song['name']} has no stem for {ROLES[what]}"
        else:
            out[source["name"]] = f"{song['name']}, {ROLES[what]}"
    return out


def feeds(args) -> None:
    """One WAV per loudspeaker: the PA's channel of the master, or a monitor mix, at the
    same gated RMS, then scaled by the source's level_db against the loudest's. A mix
    none of whose stems the song has (a song of your own given without them) is
    silence."""
    rate = pipeline.response_rate(simulations(args)[0], responses_of(args))
    song = song_of(args)
    folder = music_folder(args)
    if not (folder / "master.wav").exists():
        raise SystemExit(f"error: no music yet; run: {DEMO.command('music', args)}")

    def load(name: str) -> np.ndarray:
        x, r = read(folder / f"{name}.wav")
        x = x if x.ndim > 1 else x[:, None]
        return resampled(x, r, rate)

    master = load("master")
    if master.shape[1] < 2:
        master = np.repeat(master, 2, axis=1)
    stems = {name: load(name) for name in song["stems"]}
    length = len(master)
    if stems:
        # Line the stems up with the master: Spicy Funk Cake's start 58 ms before it (and
        # run a second longer).
        together = sum(s.mean(axis=1)[: rate * 90] for s in stems.values())
        lag = lag_of(together, master.mean(axis=1)[: rate * 90], rate, 5.0)
        stems = {k: shifted(v, lag, length) for k, v in stems.items()}
        print(f"stems line up with the master {lag / rate:+.3f} s later")

    def mix(gains: dict[str, float], others: float | None = None) -> np.ndarray | None:
        """The stems at ``gains`` (dB), the song's other stems at ``others`` or left out;
        None when the song has none of them."""
        levels = {k: others for k in stems} if others is not None else {}
        levels |= {k: g for k, g in gains.items() if k in stems}
        if not levels:
            return None
        return sum(stems[k].mean(axis=1) * 10.0 ** (g / 20.0) for k, g in levels.items())

    signals = {
        "left": master[:, 0],
        "right": master[:, 1],
        "mono": master.mean(axis=1),
    } | {name: mix(gains, others) for name, (gains, others) in MONITOR_MIXES.items()}
    signals |= {name: mix(gains) for name, gains in BAND_MIX.items()}
    # The master's channels share one gain, so its stereo balance stays, set from both
    # (their mean square): one from the mixdown, which two unlike channels lower, would
    # play them up to 3 dB over their sound power.
    pair = channels_rms(master, rate)
    reference = {"left": pair, "right": pair, "mono": gated_rms(master, rate)}
    norm = {
        k: reference[k] if k in reference else gated_rms(v, rate)
        for k, v in signals.items()
        if v is not None
    }
    sources = in_use(scene_of(args)["sources"])
    loudest = max(total_level(s.get("level_db", 94.0)) for s in sources)
    out = feeds_folder(args)
    table = {}
    for source in sources:
        what = role(source)
        level = total_level(source.get("level_db", 94.0))
        table[source["name"]] = {"plays": what, "level_db": level, "gain_db": level - loudest}
        if signals[what] is None:
            write(out / f"{source['name']}.wav", np.zeros(length), rate)
            table[source["name"]]["silent"] = True
            print(f"{source['name']:>20}: {what:<8} silent: {song['name']} has no stem for it")
            continue
        gain = 10.0 ** ((level - loudest) / 20.0) * 0.1 / norm[what]
        write(out / f"{source['name']}.wav", signals[what] * gain, rate)
        print(f"{source['name']:>20}: {what:<8} {level - loudest:+6.1f} dB")
    # The song as the script now has it (its tempo and tour may have been tuned since
    # ``music`` downloaded it): ``video`` reads them from here.
    played = {k: v for k, v in song.items() if k in SONG_KEPT}
    pipeline.write_json(
        out / "feeds.json",
        {
            "rate": rate,
            "length_s": length / rate,
            "song": played,
            "stems": sorted(song["stems"]),
            "sources": table,
        },
    )
    print(f"wrote {len(table)} feeds to {out}")


def feeds_folder(args) -> Path:
    """Where the feeds of the song a run plays go: ``feeds/<song>`` in the work folder,
    so that runs of other songs keep theirs; or the folder the arguments name instead
    (``feeds_at``: the helicopter demo's crossing keeps a table of its own there)."""
    at = getattr(args, "feeds_at", None)
    if at is not None:
        return Path(at)
    return args.work / "feeds" / name_part(song_of(args)["name"])


def feeds_done(args) -> bool:
    """Whether the feeds are there, of the song the run plays (its tempo, tour, credit
    and stems), of every source of the scene in use, at the rate of the responses and
    newer than the scene and the song's files."""
    path = feeds_folder(args) / "feeds.json"
    if not path.exists() or not args.scene.exists():
        return False
    table = pipeline.read_json(path)
    song = song_of(args)
    if table.get("song") != {k: v for k, v in song.items() if k in SONG_KEPT}:
        return False
    if table.get("stems") != sorted(song["stems"]):
        return False
    names = {source["name"] for source in in_use(scene_of(args)["sources"])}
    if set(table.get("sources", {})) != names:
        return False
    if not all((path.parent / f"{name}.wav").exists() for name in names):
        return False
    folder = music_folder(args)
    made = [args.scene, *(folder / f"{name}.wav" for name in ("master", *song["stems"]))]
    if any(p.exists() and p.stat().st_mtime > path.stat().st_mtime for p in made):
        return False
    simulation = simulations(args)[0]
    try:
        entries = pipeline.read_json(simulation.results / pipeline.PARAMETERS)["responses"]
        return table.get("rate") == pipeline.response_rate(simulation, entries)
    except (OSError, ValueError, KeyError, IndexError):
        return False


def feeds_table(args) -> dict:
    path = feeds_folder(args) / "feeds.json"
    if not path.exists():
        raise SystemExit(f"error: no feeds yet; run: {DEMO.command('feeds', args)}")
    return pipeline.read_json(path)


# --------------------------------------------------------------------------- the mixes


def render(args) -> None:
    """Every listener's mix (``--format``, binaural by default) of the sources that play
    (``--sources``: the others left out of the mixes, their feeds and their weights),
    with one gain for all: the loudest peak at -1 dBFS. The feeds carry each source's
    level_db against the loudest source's already, so auralize is not to apply it
    (``--equal-power``) and some sources keep the balance they have with them all."""
    table = feeds_table(args)
    listeners = picked(args, sorted({e["listener"] for e in responses_of(args)}))
    feeds = {name: feeds_folder(args) / f"{name}.wav" for name in table["sources"]}
    weights = {name: entry["gain_db"] for name, entry in table["sources"].items()}
    simulation = simulations(args)[0]
    mixes = [pipeline.Mix(name, simulation, name, feeds, weights) for name in listeners]
    pipeline.render_mixes(
        mixes, run_folder(args) / "seats", args, equal_power=True, inputs=args.inputs
    )


def listeners_tour(names: Sequence[str], song: dict) -> list[tuple[str, float]]:
    """The tour at the listeners ``names`` (``--listeners``) in their order: each for as
    many phrases as the song's tour gives it, else the default tour (:data:`TOUR`), else
    one, a stop the song's tour leaves out (0) included as it is named; the last runs to
    the end of the song."""
    default = {name: n for name, n in reversed(TOUR) if n > 0}  # each one's first stop
    own = song.get("tour", {})
    return [(name, own[name] if own.get(name, 0) > 0 else default.get(name, 1.0)) for name in names]


def tour_of(args, rendered) -> list[tuple[str, float, float]]:
    """(listener, start, end) in the master's time, leaving out the listeners of the
    tour that have no mix (a scene without the stage's players, say). The stops end on
    the song's phrase boundaries, half a crossfade before a downbeat, so that each
    crossfade ends on one; the first stop also takes what comes before the first. The
    tour is ``--tour``'s, at the listeners ``--listeners`` names among them; else with
    ``--listeners``, those listeners in its order (:func:`listeners_tour`); else the
    default tour, as the song changes it. Stops at one listener in a row are joined."""
    table = feeds_table(args)
    song = table["song"]
    named = listener_names(args.listeners)
    if args.tour:
        steps = [
            (part.split(":")[0], float(part.split(":")[1]) if ":" in part else 1.0)
            for part in args.tour.split(",")
        ]
    elif named is not None:
        steps = listeners_tour(named, song)
    else:
        steps = [(name, song.get("tour", {}).get(name, n)) for name, n in TOUR]
        steps = [(name, n) for name, n in steps if n > 0 or name == TOUR[-1][0]]
    kept = set(picked(args, [name for name, _ in steps]))
    steps = [(name, n) for name, n in steps if name in kept]
    if not args.tour:
        skipped = sorted({name for name, _ in steps if name not in rendered})
        if skipped:
            print(f"note: no mix for {', '.join(skipped)}; the tour goes without them")
        steps = [(name, n) for name, n in steps if name in rendered]
    if not steps and args.listeners is not None:
        raise SystemExit("error: the tour visits none of the listeners --listeners names")
    tour = music.phrase_tour(
        steps,
        table["length_s"],
        phrase_s(song),
        first_s=song["downbeat_s"] - CROSSFADE_S / 2,
    )
    return pipeline.picked_tour(tour, lambda stop: True)


def gains_of(seats: dict, tour) -> dict[str, float]:
    """The gain (dB) each listener of the tour plays at: up to the loudness of the
    loudest one (K-weighted, see :func:`pipeline.gains_of`)."""
    return pipeline.gains_of(seats, [listener for listener, _, _ in tour])


def sounding_of(args, table: dict) -> list[str]:
    """The sources heard in a run (``args`` resolved), in the feeds' order (``table``,
    feeds.json): those that play (``--sources``) whose feed is not silence."""
    fed = table["sources"]
    return [name for name in playing(args, list(fed)) if not fed[name].get("silent")]


def in_words(parts: Sequence[str]) -> str:
    """``parts`` as a list in words: ``a, b and c``."""
    return " and ".join(part for part in [", ".join(parts[:-1]), *parts[-1:]] if part)


def concert_parts(sources: Sequence[dict], sounding: Collection[str], short: bool = False) -> list:
    """What of the concert plays, in words, in the scene's order: a group of
    :data:`GROUP_WORDS` whose sources (of ``sources``) all play (``sounding``) by its
    name, ``sound system``; else its groups' that all do, then its other sources that
    do, by their names (``main arrays``, ``drum kit``). ``short``: as the thumbnail
    names them (:data:`SHORT_WORDS`)."""
    words = GROUP_WORDS | (SHORT_WORDS if short else {})

    def parts(prefix: str, members: list[dict]) -> list[str]:
        heard = [s for s in members if s["name"] in sounding]
        if not heard:
            return []
        if prefix in words and len(heard) == len(members):
            return [words[prefix]]
        out: list[str] = []
        below: dict[str, list[dict]] = {}
        for source in members:
            group = source.get("group") or ""
            rest = group[len(prefix) :].lstrip("/") if prefix else group
            if rest:
                inner = f"{prefix}/{rest.split('/')[0]}" if prefix else rest.split("/")[0]
                below.setdefault(inner, []).append(source)
            elif source["name"] in sounding:
                below.setdefault("", []).append(source)
        for inner, group in below.items():
            if inner:
                out += parts(inner, group)
            else:
                out += [source["name"].replace("_", " ") for source in group]
        return out

    return parts("", list(sources))


def heard_as(args) -> str:
    """How the mixes are heard, in words: binaurally, or in the format they take
    (``in ambix3``; the tour decodes it to the two ears)."""
    label = pipeline.output_label(args)
    return "heard binaurally" if label.startswith("binaural") else f"heard in {label}"


def heading_of(args, sources: Sequence[dict], sounding: Collection[str], toured: int) -> tuple:
    """The stills' title and the line under it: what of the concert plays
    (:func:`concert_parts`), how it is heard and where, at its listeners or at the
    ``toured`` that ``--listeners`` names."""
    where = "its listeners"
    if listener_names(getattr(args, "listeners", None)) is not None:
        where = f"{'one' if toured == 1 else toured} of its listeners"
    return (
        "A concert in a stadium, simulated",
        f"The stadium example's {in_words(concert_parts(sources, sounding))}, "
        f"{heard_as(args)} at {where}",
    )


def titles_of(sounding: Collection[str]) -> dict[str, tuple[str, str]]:
    """The stops' names and where each is (:data:`TITLES`), the guitarist between the
    side fills rather than in their sound when they do not play."""
    if any(name.startswith("side_fill") for name in sounding):
        return TITLES
    return TITLES | {"guitarist": ("On stage: the guitarist", "between the side fills")}


def video(
    args,
    run_by: tuple[Demo, object] | None = None,
    also: str = "",
    heading: tuple[str, str] | None = None,
) -> None:
    """The run's audio (``<run>.wav``), one still per stop of the tour, and its video
    (``<run>.mp4``): the stills with a scrolling spectrogram in the panel at their foot
    (ffmpeg, unless ``--no-mp4``). ``run_by``, another demo and its arguments, makes the
    mixes in the folder of ``args``'s run (its ``mix`` step; ``args`` then name that
    run's folder) and is the run the stills give; ``also`` goes after their credit and
    ``heading`` (a title and the line under it) replaces theirs."""
    from acoustic_spaces.demos.pictures import save, settings_lines

    demo, run_args = run_by or (DEMO, args)
    made_by = demo.command("render" if run_by is None else "mix", run_args)
    pictures = sibling(__file__, "stadium_pictures")
    folder = run_folder(args) / "seats"
    seats = pipeline.seats_of(folder, made_by, args)
    tour = tour_of(args, seats["seats"])
    pipeline.missing_mixes(seats, [name for name, _, _ in tour], made_by)
    gains = gains_of(seats, tour)
    stops = [pipeline.TourStop(name, start, start, stop, gains[name]) for name, start, stop in tour]
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
    results = simulations(args)[0].results
    scene, table = scene_of(args), feeds_table(args)
    sounding = sounding_of(args, table)
    used = in_use(scene["sources"])
    toured = len({name for name, _, _ in tour})
    drawing = pictures.Pictures(
        scene,
        args.scene.parent,
        pictures.levels_at(results),
        seats["seats"],
        tour,
        titles_of(sounding),
        gains,
        credit=pipeline.credited(table["song"]["credit"], seats, sounding) + also,
        hrtf=args.hrtf,
        tool=settings_lines(
            demo, run_args, scene, pipeline.read_json(results / pipeline.PARAMETERS)
        ),
        heading=heading or heading_of(args, used, sounding, toured),
        playing=sounding,
        plays=", ".join(concert_parts(used, sounding, short=True)),
        binaural=heard_as(args) == "heard binaurally",
    )
    save(drawing.thumbnail(), product(args, "_thumbnail.png"))
    pipeline.video(
        run_folder(args),
        [(name, stop - start) for name, start, stop in tour],
        drawing.still,
        sound,
        out=product(args, ".mp4"),
        no_mp4=args.no_mp4,
    )


DEMO = Demo(
    name="stadium",
    example="stadium_concert",
    title="A band in the stadium",
    description=__doc__,
    script=Path(__file__),
    steps=[
        Step(
            "music",
            music_step,
            "download the master and the stems (42 to 313 MB)",
            music_done,
            music_download,
        ),
        Step(
            "band",
            band,
            "scene_band.yaml: the scene with the band on the stage",
            lambda args: pipeline.scene_made(args.scene, args.source),
        ),
        SIMULATE,
        Step("feeds", feeds, "one signal per loudspeaker, as loud as the scene says", feeds_done),
        Step(
            "render",
            render,
            "every listener in --format (binaural by default), with one gain for all",
        ),
        Step("video", video, "the run's video, its audio and the stills, named after the run"),
    ],
    options=[
        Option(
            "song",
            "the song",
            "spicy_funk_cake",
            "choice",
            tuple(sorted(SONGS)),
            ("music", "feeds"),
            named="plays",
            replaced_by="music",
        ),
        *music.own_song_options(("music", "feeds"), stems=STEMS, downbeat=True),
        Option(
            "source",
            "the example's scene",
            Path("scene.yaml"),
            "file",
            steps=("band",),
            flag="--from",
            filter="Scene files (*.yaml *.yml)",
        ),
        Option("scene", "the scene with the band", Path("scene_band.yaml"), "path"),
        Option(
            "results",
            "folder written by simulate, a folder in it for each output",
            Path("results_band"),
            "folder",
        ),
        *common_options(),
        input_option(),
        listeners_option(help=LISTENERS_HELP),
        Option(
            "tour",
            "listener:phrases,... (8 bars each, 0.5 is 4; the last runs to the end)",
            None,
            steps=("video",),
        ),
        *sources_options(help=SOURCES_HELP, steps=("render", "video")),
    ],
    simulations=simulations,
    scene_from="band",
    plays=plays,
    problems=problems,
    # The band step keeps the example's listeners as they are.
    listeners=lambda args: listeners_in(args.source),
    sources=sources,
)


__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
