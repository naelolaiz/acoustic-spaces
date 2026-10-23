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
    acoustic-spaces simulate scene_band.yaml --out results_band --output binaural
    python stadium_demo.py feeds     # one signal per loudspeaker, as loud as the scene says
    python stadium_demo.py render    # every listener, binaural
    python stadium_demo.py video     # demo/tour.wav, the stills, demo/tour.mp4

or `python stadium_demo.py all` once the simulation is done. Everything goes to
./demo (--work); a second song goes best to a folder of its own (`--song francium
--work demo_francium`). Use headphones; `--hrtf me.sofa` on every step and on `simulate`
plays through your own or measured HRTFs instead of a spherical head.

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

import argparse
import io
import json
import math
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml

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
#: 8 bars, s. The tour moves on phrase boundaries: a crossfade that starts half a second
#: before one ends on its downbeat.
PHRASE_BARS = 8

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


# --------------------------------------------------------------------------- audio


def read(path: Path) -> tuple[np.ndarray, int]:
    from acoustic_spaces.audio import read_wav

    return read_wav(path)


def write(path: Path, data: np.ndarray, rate: int) -> None:
    from acoustic_spaces.audio import write_wav

    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, data, rate)


def gated_rms(x: np.ndarray, rate: int) -> float:
    """RMS over the 400 ms blocks that are playing: louder than -70 dBFS and than 20 dB
    below the mean of those (as loudness meters gate, without the K weighting)."""
    mono = x if x.ndim == 1 else x.mean(axis=1)
    block = int(0.4 * rate)
    n = len(mono) // block
    if n == 0:
        return float(np.sqrt(np.mean(mono**2)))
    power = (mono[: n * block].reshape(n, block) ** 2).mean(axis=1)
    on = power[power > 1e-7]
    if on.size == 0:
        return 0.0
    on = on[on > on.mean() * 0.01]
    return float(np.sqrt(on.mean()))


def a_weighted(x: np.ndarray, rate: int) -> np.ndarray:
    """``x`` (samples, channels) through the A weighting (IEC 61672), by FFT."""
    n = 1 << math.ceil(math.log2(len(x)))  # a length the FFT is fast at
    f = np.fft.rfftfreq(n, 1.0 / rate)
    f2 = f**2
    gain = (12194.0**2 * f2**2) / (
        (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
    )
    gain *= 10.0 ** (2.0 / 20.0)  # 0 dB at 1 kHz
    rows = np.ascontiguousarray(np.atleast_2d(x.T))  # a channel per row
    return np.fft.irfft(np.fft.rfft(rows, n) * gain, n)[:, : len(x)].T


def k_weighted(x: np.ndarray, rate: int) -> np.ndarray:
    """``x`` (samples, channels) through the K weighting of loudness meters (ITU-R
    BS.1770: a 4 dB shelf above 1.5 kHz and a high-pass at 38 Hz), by FFT."""
    n = 1 << math.ceil(math.log2(len(x)))
    z = np.exp(-1j * np.pi * np.fft.rfftfreq(n, 0.5))  # z^-1 at each bin
    w, a = 2.0 * np.pi * 1500.0 / rate, 10.0 ** (4.0 / 40.0)
    alpha, cos = math.sin(w) / math.sqrt(2.0), math.cos(w)
    shelf = (
        a * ((a + 1) + (a - 1) * cos + 2 * math.sqrt(a) * alpha)
        - 2 * a * ((a - 1) + (a + 1) * cos) * z
        + a * ((a + 1) + (a - 1) * cos - 2 * math.sqrt(a) * alpha) * z**2
    ) / (
        (a + 1)
        - (a - 1) * cos
        + 2 * math.sqrt(a) * alpha
        + 2 * ((a - 1) - (a + 1) * cos) * z
        + ((a + 1) - (a - 1) * cos - 2 * math.sqrt(a) * alpha) * z**2
    )
    w = 2.0 * np.pi * 38.0 / rate
    alpha, cos = math.sin(w), math.cos(w)  # Q = 0.5
    high_pass = (1 + cos) / 2 * (1 - z) ** 2 / ((1 + alpha) - 2 * cos * z + (1 - alpha) * z**2)
    rows = np.ascontiguousarray(np.atleast_2d(x.T))
    return np.fft.irfft(np.fft.rfft(rows, n) * shelf * high_pass, n)[:, : len(x)].T


def lag_of(x: np.ndarray, reference: np.ndarray, rate: int, most_s: float) -> int:
    """Samples by which ``x`` lags ``reference`` (both mono), within ``most_s``."""
    n = 1 << math.ceil(math.log2(len(x) + len(reference)))
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(reference, n)), n)
    most = int(most_s * rate)
    lags = np.r_[np.arange(0, most + 1), np.arange(-most, 0)]
    return int(lags[np.argmax(np.abs(c[lags]))])


def shifted(x: np.ndarray, samples: int, length: int) -> np.ndarray:
    """``x`` started ``samples`` earlier (later if negative), cut or padded to ``length``."""
    out = np.zeros((length,) + x.shape[1:])
    if samples >= 0:
        part = x[samples : samples + length]
        out[: len(part)] = part
    else:
        part = x[: max(0, length + samples)]
        out[-samples : -samples + len(part)] = part
    return out


# --------------------------------------------------------------------------- steps


class RangeFile(io.RawIOBase):
    """A remote file read by HTTP range requests, so that zipfile can take one member out
    of a large archive without downloading it."""

    BLOCK = 4 << 20

    def __init__(self, url: str):
        self.url, self.pos, self.cache = url, 0, {}
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            self.size = int(response.headers["Content-Range"].split("/")[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            a, b = i * self.BLOCK, min(self.size, (i + 1) * self.BLOCK) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(request, timeout=300) as response:
                self.cache[i] = response.read()
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def readinto(self, buffer) -> int:
        n = max(0, min(len(buffer), self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            block = self._block((self.pos + len(out)) // self.BLOCK)
            start = (self.pos + len(out)) % self.BLOCK
            out += block[start : start + n - len(out)]
        buffer[:n] = out
        self.pos += n
        return n


def song_of(args) -> dict:
    return SONGS[args.song] | {"name": args.song}


def phrase_s(song: dict) -> float:
    """8 bars of ``song``, s."""
    return PHRASE_BARS * 4 * 60.0 / song["bpm"]


def music(args) -> None:
    """Download ``--song``'s master and stems: only the files the song names, out of its
    release's archives or one by one, each stem's files decoded and summed. Files
    already in the folder are kept."""
    song = song_of(args)
    folder = music_folder(args)
    folder.mkdir(parents=True, exist_ok=True)
    archives: dict[str, zipfile.ZipFile] = {}
    for name, files in ({"master": song["master"]} | song["stems"]).items():
        if (folder / f"{name}.wav").exists():
            continue
        total, rate = None, 0
        for url, member in files:
            original = folder / urllib.parse.unquote(
                Path(member or urllib.parse.urlparse(url).path).name
            )
            if not original.exists():
                part = original.with_name(original.name + ".part")
                if member:
                    if url not in archives:
                        print(f"reading from {url}")
                        archives[url] = zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 20))
                    with archives[url].open(member) as source, part.open("wb") as target:
                        shutil.copyfileobj(source, target, 1 << 20)
                else:
                    print(f"downloading {url}")
                    with urllib.request.urlopen(url) as response, part.open("wb") as target:
                        shutil.copyfileobj(response, target, 1 << 20)
                part.replace(original)
            x, rate = decoded(original)
            total = x if total is None else total[: len(x)] + x[: len(total)]
        write(folder / f"{name}.wav", total, rate)
        print(f"wrote {folder / f'{name}.wav'} from {len(files)} file(s)")
    for archive in archives.values():
        archive.close()
    (folder / "SOURCES.txt").write_text(song["sources"], encoding="utf-8")
    (folder / "song.json").write_text(
        json.dumps({k: v for k, v in song.items() if k in SONG_KEPT}, indent=2) + "\n",
        encoding="utf-8",
    )


def decoded(path: Path) -> tuple[np.ndarray, int]:
    """A WAV file read, or any other decoded by ffmpeg (stereo, 32-bit float)."""
    if path.suffix.lower() == ".wav":
        return read(path)
    if shutil.which("ffmpeg") is None:
        raise SystemExit(f"error: ffmpeg is needed to decode {path.name}")
    wav = path.with_name(path.name + ".decoded.wav")
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", str(wav)]
    subprocess.run(command, check=True)
    try:
        return read(wav)
    finally:
        wav.unlink()


def music_folder(args) -> Path:
    return args.work / "music" / args.song


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
    args.scene.write_text(
        "# The stadium concert example with the band on the stage, written by stadium_demo.py.\n"
        + yaml.safe_dump(scene, sort_keys=False, default_flow_style=None, width=100),
        encoding="utf-8",
    )
    print(f"wrote {args.scene}; now: {simulate_command(args)}")


def scene_of(args) -> dict:
    return yaml.safe_load(args.scene.read_text(encoding="utf-8"))


def simulate_command(args) -> str:
    """The simulation the demo mixes from: the scene, the mixes' output and, with
    ``--hrtf``, the SOFA file its binaural responses go through."""
    command = f"acoustic-spaces simulate {args.scene} --out {args.results} --output {args.format}"
    return command + hrtf_option(args)


def hrtf_option(args) -> str:
    """`` --hrtf FILE`` for the simulate command when ``--hrtf`` names a SOFA file."""
    hrtf = getattr(args, "hrtf", None)
    return f" --hrtf {hrtf}" if hrtf else ""


def output_label(args) -> str:
    """The output ``render`` asks auralize for: ``--format``, or with ``--hrtf`` the
    binaural output simulated through that file (``binaural-<its name>``), so that mixes
    through a spherical head cannot pass for ones through the listener's own ears."""
    hrtf = getattr(args, "hrtf", None)
    if hrtf and args.format == "binaural":
        return f"binaural-{Path(hrtf).stem}"
    return args.format


def check_hrtf(args, entries: list[dict]) -> None:
    """With ``--hrtf``, stop with the simulate command to run unless the responses were
    made through that file."""
    label = output_label(args)
    if label == args.format:
        return
    labels = {o["format"] for e in entries for o in e.get("outputs") or [{"format": "mono"}]}
    if label not in labels:
        raise SystemExit(
            f"error: {args.results} has no responses through {args.hrtf}"
            f" ({', '.join(sorted(labels))}); run first:\n  {simulate_command(args)}"
        )


def responses_of(args) -> list[dict]:
    path = args.results / "parameters.json"
    if not path.exists():
        first = "" if args.scene.exists() else "  python stadium_demo.py band\n"
        raise SystemExit(
            f"error: no simulation in {args.results}; run first:\n{first}  {simulate_command(args)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))["responses"]


def response_rate(args) -> int:
    entry = responses_of(args)[0]
    outputs = entry.get("outputs") or [{"wav": entry["wav"]}]
    return read(args.results / outputs[0]["wav"])[1]


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


def feeds(args) -> None:
    """One WAV per loudspeaker: the PA's channel of the master, or a monitor mix, at the
    same gated RMS, then scaled by the source's level_db against the loudest's."""
    rate = response_rate(args)
    song = song_of(args)
    folder = music_folder(args)
    if not (folder / "master.wav").exists():
        raise SystemExit(
            f"error: no music yet; run: python stadium_demo.py music --song {args.song}"
        )
    from acoustic_spaces.render import resample

    def load(name: str) -> np.ndarray:
        x, r = read(folder / f"{name}.wav")
        x = x if x.ndim > 1 else x[:, None]
        return resample(x, r, rate) if r != rate else x

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
    (out / "feeds.json").write_text(
        json.dumps(
            {"rate": rate, "length_s": length / rate, "song": played, "sources": table}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(table)} feeds to {out}")


def arrival_s(args, listener: str, table: dict) -> float:
    """When the loudest sound reaches ``listener`` (s): the peak, over 2 ms, of the energy
    of its responses, each weighted by how loud its source plays. Cross-correlating the
    mix with the master would be simpler, but the song's beat repeats, and on the stands
    that picked a beat 0.25 s late."""
    total, rate = np.zeros(0), 1
    for entry in responses_of(args):
        if entry["listener"] != listener or entry["source"] not in table["sources"]:
            continue
        outputs = entry.get("outputs") or [{"wav": entry["wav"]}]
        h, rate = read(args.results / outputs[0]["wav"])
        energy = (h**2).sum(axis=1) if h.ndim > 1 else h**2
        energy *= 10.0 ** (table["sources"][entry["source"]]["gain_db"] / 10.0)
        total = np.pad(total, (0, max(0, len(energy) - len(total))))
        total[: len(energy)] += energy
    smooth = np.convolve(total, np.ones(max(1, int(0.002 * rate))), "same")
    return int(np.argmax(smooth)) / rate


def render(args) -> None:
    """Every listener's mix (``--format``, binaural by default), with one gain for all:
    the loudest peak at -1 dBFS."""
    from acoustic_spaces.cli import main as cli

    table = json.loads((args.work / "feeds" / "feeds.json").read_text(encoding="utf-8"))
    rate = table["rate"]
    entries = responses_of(args)
    check_hrtf(args, entries)
    listeners = sorted({e["listener"] for e in entries})
    if args.listeners:
        listeners = [name for name in args.listeners.split(",")]
    pairs = [f"{name}={args.work / 'feeds' / f'{name}.wav'}" for name in table["sources"]]
    seats = args.work / "seats"
    peaks = {}
    for listener in listeners:
        out = seats / f"{listener}.{args.format}.wav"
        # The feeds carry each source's level_db already: auralize is not to apply it.
        command = ["auralize", str(args.results), *pairs, "--listener", listener]
        code = cli(command + ["--format", output_label(args), "--out", str(out), "--equal-power"])
        if code:
            raise SystemExit(code)
        peaks[listener] = float(np.abs(read(out)[0]).max())
    gain = 10.0 ** (-1.0 / 20.0) / max(peaks.values())
    info = {}
    for listener in listeners:
        out = seats / f"{listener}.{args.format}.wav"
        x = read(out)[0] * gain
        write(out, x, rate)
        level = gated_rms(a_weighted(x, rate), rate)
        loudness = gated_rms(k_weighted(x, rate), rate)
        info[listener] = {
            "lag_s": arrival_s(args, listener, table),
            "rms_db": 20.0 * math.log10(level + 1e-12),
            "loudness_db": 20.0 * math.log10(loudness + 1e-12),
        }
    top = max(v["rms_db"] for v in info.values())
    for listener, v in info.items():
        v["relative_db"] = v["rms_db"] - top
        late = v["lag_s"] * 1e3
        print(f"{listener:>20}: {v['relative_db']:+6.1f} dB(A), sound arrives {late:5.0f} ms late")
    (seats / "seats.json").write_text(
        json.dumps({"rate": rate, "format": args.format, "gain": gain, "seats": info}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def tour_of(args, rendered) -> list[tuple[str, float, float]]:
    """(listener, start, end) in the master's time, leaving out the listeners of the
    default tour that have no mix (a scene without the stage's players, say). The stops
    end on the song's phrase boundaries, half a crossfade before a downbeat, so that each
    crossfade ends on one; the first stop also takes what comes before the first."""
    table = json.loads((args.work / "feeds" / "feeds.json").read_text(encoding="utf-8"))
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
    end = table["length_s"]
    first = song["downbeat_s"] - CROSSFADE_S / 2
    tour, t, phrases = [], 0.0, 0.0
    for k, (listener, n) in enumerate(steps):
        phrases += n
        stop = end if k == len(steps) - 1 or n <= 0 else min(end, first + phrases * phrase_s(song))
        if stop > t:
            tour.append((listener, t, stop))
        t = stop
    return tour


def gains_of(seats: dict, tour) -> dict[str, float]:
    """The gain (dB) each listener of the tour plays at: up to the loudness of the
    loudest one. Loudness as meters weigh it (K), not the A weighting of the levels: the
    subwoofers' bass, which the A weighting all but leaves out, is heard."""
    levels = {}
    for listener, _, _ in tour:
        seat = seats["seats"][listener]
        levels[listener] = seat.get("loudness_db", seat["rms_db"])
    top = max(levels.values())
    return {listener: top - level for listener, level in levels.items()}


def tour_audio(args, tour, seats) -> np.ndarray:
    rate = seats["rate"]
    length = round(tour[-1][2] * rate)
    out = np.zeros((length, 2))
    half = int(CROSSFADE_S * rate / 2)
    gains = gains_of(seats, tour)
    for k, (listener, start, stop) in enumerate(tour):
        x = read(args.work / "seats" / f"{listener}.{seats['format']}.wav")[0]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x = shifted(x, round(seats["seats"][listener]["lag_s"] * rate), length)
        x = x * 10.0 ** (gains[listener] / 20.0)
        a, b = round(start * rate), round(stop * rate)
        w = np.zeros(length)
        lo, hi = (a - half if k else 0), (b + half if k < len(tour) - 1 else length)
        w[lo:hi] = 1.0
        ramp = np.sin(np.linspace(0.0, np.pi / 2, 2 * half))  # equal power
        if k:
            w[a - half : a + half] = ramp
        if k < len(tour) - 1:
            w[b - half : b + half] = ramp[::-1]
        out += x[:, :2] * w[:, None]
    fade = int(FADE_OUT_S * rate)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None] ** 2
    peak = float(np.abs(out).max())
    return out * (10.0 ** (-1.0 / 20.0) / peak if peak > 0 else 1.0)


def video(args) -> None:
    """tour.wav, one still per stop of the tour, and tour.mp4 with a scrolling
    spectrogram under the stills (ffmpeg)."""
    import stadium_pictures as pictures

    seats = json.loads((args.work / "seats" / "seats.json").read_text(encoding="utf-8"))
    tour = tour_of(args, seats["seats"])
    missing = [name for name, _, _ in tour if name not in seats["seats"]]
    if missing:
        raise SystemExit(f"error: no mix for {missing[0]}; run: python stadium_demo.py render")
    audio = tour_audio(args, tour, seats)
    out = args.work
    write(out / "tour.wav", audio, seats["rate"])
    print(f"wrote {out / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    levels = pictures.levels_at(args.results)
    table = json.loads((args.work / "feeds" / "feeds.json").read_text(encoding="utf-8"))
    drawing = pictures.Pictures(
        scene_of(args),
        args.scene.parent,
        levels,
        seats["seats"],
        tour,
        TITLES,
        gains_of(seats, tour),
        credit=table["song"]["credit"],
    )
    stills = out / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, (listener, start, stop) in enumerate(tour):
        path = stills / f"{k:02d}_{listener}.png"
        drawing.still(k).convert("RGB").save(path, optimize=True)
        lines += [f"file '{path.resolve().as_posix()}'", f"duration {stop - start:.3f}"]
    lines.append(lines[-2])  # the concat demuxer needs the last picture again
    (stills / "stills.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    drawing.thumbnail().convert("RGB").save(out / "thumbnail.png", optimize=True)
    print(f"wrote {len(tour)} stills to {stills} and {out / 'thumbnail.png'}")
    if args.no_mp4:
        return
    if shutil.which("ffmpeg") is None:
        raise SystemExit("error: ffmpeg is needed for the video")
    band = pictures.SPECTRUM_BOX
    spectrum = (
        f"[1:a]showspectrum=s={band[2]}x512:slide=scroll:color=magma:scale=log:fscale=log:"
        f"start=40:stop=16000:drange=80:overlap=0.5:fps=30,scale={band[2]}:{band[3]},"
        "format=rgba,colorkey=0x000000:0.08:0.25[s];"
        f"[0:v]fps=30,format=rgba[b];[b][s]overlay={band[0]}:{band[1]}:shortest=1,format=yuv420p[v]"
    )
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-stats", "-y",
            "-f", "concat", "-safe", "0", "-i", str(stills / "stills.txt"),
            "-i", str(out / "tour.wav"),
            "-filter_complex", spectrum, "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-c:a", "aac", "-b:a", "384k", "-movflags", "+faststart",
            str(out / "tour.mp4"),
        ],
        check=True,
    )  # fmt: skip
    print(f"wrote {out / 'tour.mp4'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["music", "band", "feeds", "render", "video", "all"])
    parser.add_argument(
        "--song",
        choices=sorted(SONGS),
        default="spicy_funk_cake",
        help="music, feeds: the song (default spicy_funk_cake)",
    )
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        default=Path("scene.yaml"),
        help="band: the example's scene",
    )
    parser.add_argument(
        "--scene", type=Path, default=Path("scene_band.yaml"), help="the scene with the band"
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results_band"), help="folder written by simulate"
    )
    parser.add_argument(
        "--work", type=Path, default=Path("demo"), help="folder for everything this writes"
    )
    parser.add_argument(
        "--format", default="binaural", help="output format the mixes take (default binaural)"
    )
    parser.add_argument(
        "--hrtf",
        type=Path,
        help="binaural mixes through this SOFA file of HRTFs, your own or measured, instead of "
        "a spherical head model (the simulate command the script prints takes it too)",
    )
    parser.add_argument("--listeners", help="render: only these listeners, comma separated")
    parser.add_argument(
        "--tour",
        help="video: listener:phrases,... (8 bars each, 0.5 is 4; the last runs to the end)",
    )
    parser.add_argument(
        "--no-mp4", action="store_true", help="video: write the audio and stills only"
    )
    args = parser.parse_args()
    if args.hrtf and args.format != "binaural":
        parser.error("--hrtf is for binaural mixes")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    steps = ["music", "feeds", "render", "video"] if args.step == "all" else [args.step]
    for step in steps:
        globals()[step](args)


if __name__ == "__main__":
    main()
