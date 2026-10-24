"""A song played by a band on the stadium example's stage and through its sound system,
heard at each of its listeners, and a video that tours them.

The song (--song) is "Spicy Funk Cake" by the Rhythmussportgruppe (composed by Niklas
Dahlheimer), recorded for the sound database of the DEGA (German Acoustical Society) at
Hochschule Düsseldorf, CC BY 4.0, https://doi.org/10.5281/zenodo.3601032, or "Francium"
or "Rubidium" by Software-Entwicklungskit from the album "Everything is Free", CC0 1.0,
https://github.com/ichbinsoftware/everythingisfree. The sound system (main arrays,
subwoofers, fills, towers and delays) plays the song's stereo master; the monitors play
mixes of its stems: the singer's wedges the vocal, the drum fill the drums and bass, the
side fills a mix for the whole stage; and the band on the stage plays its own parts: the
drum kit the drums, the singer the vocal, the guitarist's amplifier the guitar (the synth
lead, for the album's songs). Run it from the folder of the stadium example
(`acoustic-spaces examples --copy . stadium_concert && cd stadium_concert`), in the
Python environment acoustic-spaces is installed in, with ffmpeg on the PATH:

    python stadium_demo.py music     # download the master and the stems (0.4 to 0.8 GB)
    python stadium_demo.py band      # scene_band.yaml: the scene with the band on the stage
    python stadium_demo.py simulate  # simulate scene_band.yaml into results_band, binaural
    python stadium_demo.py feeds     # one signal per loudspeaker, as loud as the scene says
    python stadium_demo.py render    # every listener, binaural
    python stadium_demo.py video     # demo/tour.wav, the stills, demo/tour.mp4

or `python stadium_demo.py all`, which runs them in turn and leaves out what is there
already (the music, the scene, the simulation). The scene and its simulation go beside
the script (scene_band.yaml, results_band); the music, the feeds, the mixes and the
video go to ./demo (--work), and a second song goes best to a work folder of its own
(`--song francium --work demo_francium`).
Use headphones; `--hrtf me.sofa` plays through your own or measured HRTFs instead of a
spherical head (the simulation takes it too). `--input drum_kit=drums.wav` has a source
play a file of yours instead of its feed, at the level of that feed (a name the scene
has no source for is refused with the list of them). The editor runs the same steps
from Simulation > Run demo…

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
that time and the level there against the loudest listener's. On stage it is 7 to 20
dB(A) louder than in the audience, too much to hear both in one video, so each stop is
turned up to the loudness of the loudest one (K-weighted, as loudness meters weigh it)
and its caption says by how much; the mixes in demo/seats keep one gain for all.
"""

from __future__ import annotations

import math
import sys
import urllib.parse
from pathlib import Path

import numpy as np
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
from acoustic_spaces.demos.audio import gated_rms, lag_of, read, resampled, shifted, write

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
        bpm=132.0,
        downbeat_s=0.0,  # the master starts on its first downbeat
        tour={},
    ),
}
#: What ``music`` writes of a song to song.json and ``feeds`` to feeds.json.
SONG_KEPT = ("name", "bpm", "downbeat_s", "tour", "credit")

#: Monitor mixes: stem and its gain in dB (stems a song does not have are left out).
SINGER_MIX = {"vocal": 0, "backing": -6, "keys": -8, "guitar": -12, "bass": -14, "drums": -16}
DRUMMER_MIX = {"drums": 0, "bass": 0, "vocal": -6, "guitar": -8, "keys": -8}
# The side fills carry the song for the whole stage, with the drums well down: the kit
# itself is loud on stage, so the players hear the drums from where the drummer is.
STAGE_MIX = {"vocal": 3, "drums": -20}  # every other stem at 0 dB
MONITORS = {"vocal_wedges": "singer", "drum_fill": "drummer", "side_fill": "stage"}
#: The band itself, as sources on the stage (added by ``band``): sound power (dB re 1 pW)
#: of a rock drum kit, a singer singing loud and an instrument amplifier turned up.
BAND = {
    "drum_kit": {"position": [-52.3, 0.0, 2.8], "level_db": 116.0, "directivity": "omni"},
    "singer_voice": {
        "position": [-44.6, 0.0, 3.7],
        "orientation": {"yaw": 0.0},
        "level_db": 98.0,
        "directivity": "subcardioid",
    },
    "guitar_amp": {
        "position": [-49.0, 8.0, 2.5],
        "orientation": {"yaw": 0.0, "pitch": 20.0},
        "level_db": 112.0,
        "directivity": "cardioid",
    },
}
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
CROSSFADE_S = 1.0
FADE_OUT_S = 4.0


# --------------------------------------------------------------------------- the music


def song_of(args) -> dict:
    return SONGS[args.song] | {"name": args.song}


def phrase_s(song: dict) -> float:
    """A phrase of ``song`` (8 bars), s. The tour moves on phrase boundaries: a crossfade
    that starts half a second before one ends on its downbeat."""
    return music.phrase_s(song)


def music_folder(args) -> Path:
    return args.work / "music" / args.song


def music_done(args) -> bool:
    """Whether the song's master and stems are in the folder."""
    folder = music_folder(args)
    return all((folder / f"{name}.wav").exists() for name in ("master", *song_of(args)["stems"]))


def music_step(args) -> None:
    """Download ``--song``'s master and stems: only the files the song names, out of its
    release's archives or one by one, each stem's files decoded and summed. Files
    already in the folder are kept."""
    song = song_of(args)
    folder = music_folder(args)
    music.write_tracks({"master": song["master"]} | song["stems"], folder)
    (folder / "SOURCES.txt").write_text(song["sources"], encoding="utf-8")
    pipeline.write_json(folder / "song.json", {k: v for k, v in song.items() if k in SONG_KEPT})


# --------------------------------------------------------------------------- the scene


def band(args) -> None:
    """``--scene`` (default scene_band.yaml) written from ``--from`` (scene.yaml) with the
    band on the stage: the drum kit, the singer's voice and the guitarist's amplifier,
    so that the players are heard from where they stand and not only through the
    loudspeakers."""
    scene = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    names = {s["name"] for s in scene["sources"]}
    scene["sources"] += [
        {"name": name, "group": "Band"} | spec for name, spec in BAND.items() if name not in names
    ]
    pipeline.write_scene(
        args.scene,
        scene,
        "# The stadium concert example with the band on the stage, written by stadium_demo.py.\n",
        args.source.parent,
    )
    print(f"wrote {args.scene}; now: {pipeline.simulate_command(simulations(args)[0], args.hrtf)}")


def scene_of(args) -> dict:
    return yaml.safe_load(args.scene.read_text(encoding="utf-8"))


def simulations(args) -> list[pipeline.Simulation]:
    """The one simulation the demo mixes from: the scene with the band, the mixes' output."""
    return [pipeline.Simulation(args.scene, args.results, args.format)]


def responses_of(args) -> list[dict]:
    return pipeline.responses_of(simulations(args)[0], args, DEMO.command("band", args))


# --------------------------------------------------------------------------- the feeds


def total_level(level: float | list[float]) -> float:
    if isinstance(level, list):
        return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in level))
    return float(level)


def role(source: dict) -> str:
    """What a source plays: a monitor mix, the master in mono (subwoofers and
    loudspeakers on the stage's axis) or the master's left or right channel, as the
    audience sees it (facing the stage, which is at -x: left is -y)."""
    if source["name"] in BAND:
        return source["name"]
    for prefix, mix in MONITORS.items():
        if source["name"].startswith(prefix):
            return mix
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
    """What each source plays unless ``--input`` gives it a file (:func:`role`): the
    sources of the scene with the band, or, before ``band`` writes it, of ``--from`` and
    the band."""
    if args.scene.exists():
        sources = scene_of(args)["sources"]
    elif args.source.exists():
        sources = yaml.safe_load(args.source.read_text(encoding="utf-8"))["sources"]
        names = {s["name"] for s in sources}
        sources += [{"name": name} | spec for name, spec in BAND.items() if name not in names]
    else:
        return {}
    out = {}
    for source in sources:
        what = role(source)
        if what in BAND_MIX:
            out[source["name"]] = f"{args.song}, the {', '.join(BAND_MIX[what])} stem"
        else:
            out[source["name"]] = f"{args.song}, {ROLES[what]}"
    return out


def feeds(args) -> None:
    """One WAV per loudspeaker: the PA's channel of the master, or a monitor mix, at the
    same gated RMS, then scaled by the source's level_db against the loudest's."""
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
    stems = {name: load(name) for name in song["stems"]}
    length = len(master)
    # Line the stems up with the master: Spicy Funk Cake's start 58 ms before it (and run
    # a second longer).
    together = sum(s.mean(axis=1)[: rate * 90] for s in stems.values())
    lag = lag_of(together, master.mean(axis=1)[: rate * 90], rate, 5.0)
    stems = {k: shifted(v, lag, length) for k, v in stems.items()}
    print(f"stems line up with the master {lag / rate:+.3f} s later")

    def mix(gains: dict[str, float], others: float | None = None) -> np.ndarray:
        """The stems at ``gains`` (dB), the song's other stems at ``others`` or left out."""
        levels = {k: others for k in stems} if others is not None else {}
        levels |= {k: g for k, g in gains.items() if k in stems}
        return sum(stems[k].mean(axis=1) * 10.0 ** (g / 20.0) for k, g in levels.items())

    signals = {
        "left": master[:, 0],
        "right": master[:, 1],
        "mono": master.mean(axis=1),
        "singer": mix(SINGER_MIX),
        "drummer": mix(DRUMMER_MIX),
        "stage": mix(STAGE_MIX, others=0.0),
    } | {name: mix(gains) for name, gains in BAND_MIX.items()}
    # The master's channels share one gain, so its stereo balance stays.
    reference = gated_rms(master, rate)
    norm = {
        k: reference if k in ("left", "right", "mono") else gated_rms(v, rate)
        for k, v in signals.items()
    }
    sources = scene_of(args)["sources"]
    loudest = max(total_level(s.get("level_db", 94.0)) for s in sources)
    out = args.work / "feeds"
    table = {}
    for source in sources:
        what = role(source)
        level = total_level(source.get("level_db", 94.0))
        gain = 10.0 ** ((level - loudest) / 20.0) * 0.1 / norm[what]
        write(out / f"{source['name']}.wav", signals[what] * gain, rate)
        table[source["name"]] = {"plays": what, "level_db": level, "gain_db": level - loudest}
        print(f"{source['name']:>20}: {what:<8} {level - loudest:+6.1f} dB")
    # The song as the script now has it (its tempo and tour may have been tuned since
    # ``music`` downloaded it): ``video`` reads them from here.
    played = {k: v for k, v in song.items() if k in SONG_KEPT}
    pipeline.write_json(
        out / "feeds.json",
        {"rate": rate, "length_s": length / rate, "song": played, "sources": table},
    )
    print(f"wrote {len(table)} feeds to {out}")


def feeds_table(args) -> dict:
    path = args.work / "feeds" / "feeds.json"
    if not path.exists():
        raise SystemExit(f"error: no feeds yet; run: {DEMO.command('feeds', args)}")
    return pipeline.read_json(path)


# --------------------------------------------------------------------------- the mixes


def render(args) -> None:
    """Every listener's mix (``--format``, binaural by default), with one gain for all:
    the loudest peak at -1 dBFS. The feeds carry each source's level_db already, so
    auralize is not to apply it (``--equal-power``)."""
    table = feeds_table(args)
    listeners = sorted({e["listener"] for e in responses_of(args)})
    if args.listeners:
        listeners = args.listeners.split(",")
    feeds = {name: args.work / "feeds" / f"{name}.wav" for name in table["sources"]}
    weights = {name: entry["gain_db"] for name, entry in table["sources"].items()}
    simulation = simulations(args)[0]
    mixes = [pipeline.Mix(name, simulation, name, feeds, weights) for name in listeners]
    pipeline.render_mixes(mixes, args.work / "seats", args, equal_power=True, inputs=args.inputs)


def tour_of(args, rendered) -> list[tuple[str, float, float]]:
    """(listener, start, end) in the master's time, leaving out the listeners of the
    default tour that have no mix (a scene without the stage's players, say). The stops
    end on the song's phrase boundaries, half a crossfade before a downbeat, so that each
    crossfade ends on one; the first stop also takes what comes before the first."""
    table = feeds_table(args)
    song = table["song"]
    if args.tour:
        steps = [
            (part.split(":")[0], float(part.split(":")[1]) if ":" in part else 1.0)
            for part in args.tour.split(",")
        ]
    else:
        steps = [(name, song.get("tour", {}).get(name, n)) for name, n in TOUR]
        steps = [(name, n) for name, n in steps if n > 0 or name == TOUR[-1][0]]
        skipped = sorted({name for name, _ in steps if name not in rendered})
        if skipped:
            print(f"note: no mix for {', '.join(skipped)}; the tour goes without them")
        steps = [(name, n) for name, n in steps if name in rendered]
    return music.phrase_tour(
        steps,
        table["length_s"],
        phrase_s(song),
        first_s=song["downbeat_s"] - CROSSFADE_S / 2,
    )


def gains_of(seats: dict, tour) -> dict[str, float]:
    """The gain (dB) each listener of the tour plays at: up to the loudness of the
    loudest one (K-weighted, see :func:`pipeline.gains_of`)."""
    return pipeline.gains_of(seats, [listener for listener, _, _ in tour])


def video(args) -> None:
    """tour.wav, one still per stop of the tour, and tour.mp4: the stills with a
    scrolling spectrogram in the panel at their foot (ffmpeg, unless ``--no-mp4``)."""
    from acoustic_spaces.demos.pictures import save, settings_lines

    pictures = sibling(__file__, "stadium_pictures")
    seats = pipeline.seats_of(args.work / "seats", DEMO.command("render", args))
    tour = tour_of(args, seats["seats"])
    pipeline.missing_mixes(seats, [name for name, _, _ in tour], DEMO.command("render", args))
    gains = gains_of(seats, tour)
    stops = [pipeline.TourStop(name, start, start, stop, gains[name]) for name, start, stop in tour]
    audio = pipeline.tour_audio(
        stops, args.work / "seats", seats, crossfade_s=CROSSFADE_S, fade_out_s=FADE_OUT_S
    )
    write(args.work / "tour.wav", audio, seats["rate"])
    print(f"wrote {args.work / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    drawing = pictures.Pictures(
        scene_of(args),
        args.scene.parent,
        pictures.levels_at(args.results),
        seats["seats"],
        tour,
        TITLES,
        gains,
        credit=pipeline.credited(
            feeds_table(args)["song"]["credit"], seats, list(feeds_table(args)["sources"])
        ),
        hrtf=args.hrtf,
        tool=settings_lines(
            DEMO, args, scene_of(args), pipeline.read_json(args.results / pipeline.PARAMETERS)
        ),
    )
    save(drawing.thumbnail(), args.work / "thumbnail.png")
    pipeline.video(
        args.work,
        [(name, stop - start) for name, start, stop in tour],
        drawing.still,
        args.work / "tour.wav",
        no_mp4=args.no_mp4,
    )


DEMO = Demo(
    name="stadium",
    example="stadium_concert",
    title="A band in the stadium",
    description=__doc__,
    script=Path(__file__),
    steps=[
        Step("music", music_step, "download the master and the stems (0.4 to 0.8 GB)", music_done),
        Step(
            "band",
            band,
            "scene_band.yaml: the scene with the band on the stage",
            lambda args: args.scene.exists(),
        ),
        SIMULATE,
        Step("feeds", feeds, "one signal per loudspeaker, as loud as the scene says"),
        Step("render", render, "every listener, binaural, with one gain for all"),
        Step("video", video, "tour.wav, the stills and tour.mp4 in the work folder"),
    ],
    options=[
        Option(
            "song",
            "the song",
            "spicy_funk_cake",
            "choice",
            tuple(sorted(SONGS)),
            ("music", "feeds"),
        ),
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
        Option("results", "folder written by simulate", Path("results_band"), "folder"),
        *common_options(),
        input_option(),
        Option("listeners", "only these listeners, comma separated", None, steps=("render",)),
        Option(
            "tour",
            "listener:phrases,... (8 bars each, 0.5 is 4; the last runs to the end)",
            None,
            steps=("video",),
        ),
    ],
    simulations=simulations,
    scene_from="band",
    plays=plays,
)


__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
