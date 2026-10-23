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

    python lakeside_demo.py music    # download the song, a signal per loudspeaker
    python lakeside_demo.py scenes   # scene_demo.yaml, and scene_still.yaml in still air
    acoustic-spaces simulate scene_demo.yaml --out results_demo --output binaural
    acoustic-spaces simulate scene_still.yaml --out results_still --output binaural
    python lakeside_demo.py render   # every listener in both, binaural
    python lakeside_demo.py video    # demo/tour.wav, the stills, demo/tour.mp4

or `python lakeside_demo.py all` once the simulations are done. Everything goes to
./demo (--work). Use headphones; `--hrtf me.sofa` on every step and on `simulate` plays
through your own or measured HRTFs instead of a spherical head.

From the audience to the far shore the sound drops by 20 to 25 dB(A), too much to
hear the far places in a video. So each stop of the tour is brought to the same loudness
(A-weighted, gated) as the audience's, and its caption says how much quieter it really
is. The window in still air keeps the gain of the window in the evening's weather, so
the difference the weather makes is heard as the simulation gives it. Each place's
mix is shifted earlier by the time the sound takes to reach it (1.6 s across the
lake), so that the beat carries on across the cuts.
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

#: The songs the demo can play: where each one is (a file, or a member of a zip file
#: read by HTTP range requests), its tempo (beats a minute; the tour moves on every 8
#: bars from the start) and the credit its licence asks for.
SONGS = {
    "spicy_funk_cake": {
        "url": "https://zenodo.org/records/3601032/files/Rhythmussportgruppe_Preview.zip",
        "member": "Rhythmussportgruppe_Preview/Spicy_Fun_cake_Preview/1_REFERENCE-MIX/"
        "Spicy_Funk_Cake.wav",
        "bpm": 110.0,
        "credit": 'Music: Rhythmussportgruppe, "Spicy Funk Cake" (Niklas Dahlheimer), recorded '
        "for the DEGA at Hochschule Düsseldorf, CC BY 4.0 · doi.org/10.5281/zenodo.3601032",
    },
    "great_british_summer": {
        "url": "https://zenodo.org/records/19683001/files/masters_stereo.zip",
        "member": "masters_stereo/09 - Great British Summer.wav",
        "bpm": 120.0,
        "credit": 'Music: Tom McKenzie, "Great British Summer" (album "Extraordinary", AlbumDB), '
        "CC BY 4.0 · doi.org/10.5281/zenodo.19683001",
    },
    "potassium": {
        "url": "https://raw.githubusercontent.com/ichbinsoftware/everythingisfree/"
        "e1d0d5c0422d2626c211d5d877e6f69c37ce8712/src/4.Potassium/4.Potassium_Master.m4a",
        "bpm": 90.0,
        "credit": 'Music: Software-Entwicklungskit, "Potassium" (album "Everything is Free"), '
        "CC0 1.0 · github.com/ichbinsoftware/everythingisfree",
    },
}
#: The rate the songs are brought to, that of the simulation's responses.
RATE = 48000

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
    on = power[power > 1e-7 * power.max()]
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
    return 8 * 4 * 60.0 / song["bpm"]


def music(args) -> None:
    """Download ``--song`` (from a zip file, only its own member), bring it to 48 kHz with
    ffmpeg, and write what each loudspeaker plays (SYSTEM, "channel"). A song already
    in the folder is kept."""
    song = song_of(args)
    folder = args.work / "music"
    folder.mkdir(parents=True, exist_ok=True)
    wav = folder / f"{args.song}.wav"
    if not wav.exists():
        if shutil.which("ffmpeg") is None:
            raise SystemExit("error: ffmpeg is needed to decode the song")
        url = song["url"]
        original = folder / (Path(song.get("member") or urllib.parse.urlparse(url).path).name)
        if not original.exists():
            print(f"downloading {url}" + (f" ({song['member']})" if "member" in song else ""))
            if "member" in song:
                with zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 20)) as archive:
                    original.write_bytes(archive.read(song["member"]))
            else:
                with urllib.request.urlopen(url) as response:
                    original.write_bytes(response.read())
        command = ["ffmpeg", "-v", "error", "-y", "-i", str(original), "-ac", "2"]
        command += ["-ar", str(RATE), "-c:a", "pcm_f32le", str(wav)]
        subprocess.run(command, check=True)
        print(f"wrote {wav}")
    x, rate = read(wav)
    for name, source in SYSTEM.items():
        channel = source["channel"]
        write(folder / f"{name}.wav", x.mean(axis=1) if channel == "mono" else x[:, channel], rate)
    (folder / "song.json").write_text(json.dumps(song, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {', '.join(SYSTEM)} ({len(x) / rate:.0f} s of {args.song}) to {folder}")


def played(args) -> dict:
    """The song ``music`` last wrote the loudspeakers' signals from."""
    path = args.work / "music" / "song.json"
    if not path.exists():
        raise SystemExit("error: no song yet; run first: python lakeside_demo.py music")
    return json.loads(path.read_text(encoding="utf-8"))


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
        path.write_text(
            f"# The lakeside concert example {what}, written by lakeside_demo.py.\n"
            + yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=100),
            encoding="utf-8",
        )
    print(f"wrote {args.scene} and {args.still}; now:")
    for scene_path, results in ((args.scene, args.results), (args.still, args.still_results)):
        print(f"  {simulate_command(args, scene_path, results)}")


def simulate_command(args, scene: Path, results: Path) -> str:
    """The simulation the demo mixes from: a weather's scene, the mixes' output and, with
    ``--hrtf``, the SOFA file its binaural responses go through."""
    command = f"acoustic-spaces simulate {scene} --out {results} --output {args.format}"
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


def check_hrtf(args, weather: str, entries: list[dict]) -> None:
    """With ``--hrtf``, stop with the simulate command to run unless the responses of
    ``weather`` were made through that file."""
    label = output_label(args)
    if label == args.format:
        return
    labels = {o["format"] for e in entries for o in e.get("outputs") or [{"format": "mono"}]}
    if label not in labels:
        scene, results = weathers(args)[weather]
        raise SystemExit(
            f"error: {results} has no responses through {args.hrtf}"
            f" ({', '.join(sorted(labels))}); run first:\n"
            f"  {simulate_command(args, scene, results)}"
        )


def weathers(args) -> dict[str, tuple[Path, Path]]:
    """Each weather's scene and results folder."""
    return {
        "evening": (args.scene, args.results),
        "still": (args.still, args.still_results),
    }


def responses_of(args, weather: str) -> list[dict]:
    scene, results = weathers(args)[weather]
    path = results / "parameters.json"
    if not path.exists():
        first = "" if scene.exists() else "  python lakeside_demo.py scenes\n"
        raise SystemExit(
            f"error: no simulation in {results}; run first:\n"
            f"{first}  {simulate_command(args, scene, results)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))["responses"]


def arrival_s(args, weather: str, listener: str) -> float:
    """When the loudest sound reaches ``listener`` (s): the peak, over 2 ms, of the
    energy of its responses from every loudspeaker."""
    total, rate = np.zeros(0), 1
    results = weathers(args)[weather][1]
    for entry in responses_of(args, weather):
        if entry["listener"] != listener:
            continue
        outputs = entry.get("outputs") or [{"wav": entry["wav"]}]
        h, rate = read(results / outputs[0]["wav"])
        energy = (h**2).sum(axis=1) if h.ndim > 1 else h**2
        total = np.pad(total, (0, max(0, len(energy) - len(total))))
        total[: len(energy)] += energy
    smooth = np.convolve(total, np.ones(max(1, int(0.002 * rate))), "same")
    return int(np.argmax(smooth)) / rate


def render(args) -> None:
    """Every listener's mix in both weathers (``--format``, binaural by default), with one
    gain for all: the loudest peak at -1 dBFS; and each one's level and delay."""
    from acoustic_spaces.cli import main as cli

    music_dir = args.work / "music"
    pairs = [f"{name}={music_dir / f'{name}.wav'}" for name in SYSTEM]
    seats = args.work / "seats"
    peaks, rate = {}, 0
    for weather in weathers(args):
        results = weathers(args)[weather][1]
        entries = responses_of(args, weather)
        check_hrtf(args, weather, entries)
        listeners = sorted({e["listener"] for e in entries})
        for listener in listeners:
            out = seats / f"{listener}.{weather}.{args.format}.wav"
            command = ["auralize", str(results), *pairs, "--listener", listener]
            code = cli(command + ["--format", output_label(args), "--out", str(out)])
            if code:
                raise SystemExit(code)
            x, rate = read(out)
            peaks[(listener, weather)] = float(np.abs(x).max())
    gain = 10.0 ** (-1.0 / 20.0) / max(peaks.values())
    info: dict = {}
    for listener, weather in peaks:
        out = seats / f"{listener}.{weather}.{args.format}.wav"
        x = read(out)[0] * gain
        write(out, x, rate)
        level = gated_rms(a_weighted(x, rate), rate)
        loudness = gated_rms(k_weighted(x, rate), rate)
        info[f"{listener}.{weather}"] = {
            "listener": listener,
            "weather": weather,
            "lag_s": arrival_s(args, weather, listener),
            "rms_db": 20.0 * math.log10(level + 1e-12),
            "loudness_db": 20.0 * math.log10(loudness + 1e-12),
        }
    top = max(v["rms_db"] for v in info.values())
    for key, v in sorted(info.items(), key=lambda kv: -kv[1]["rms_db"]):
        v["relative_db"] = v["rms_db"] - top
        late = v["lag_s"] * 1e3
        print(f"{key:>24}: {v['relative_db']:+6.1f} dB(A), sound arrives {late:5.0f} ms late")
    (seats / "seats.json").write_text(
        json.dumps({"rate": rate, "format": args.format, "gain": gain, "seats": info}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def tour_of(args) -> list[tuple[str, str, float, float]]:
    """(listener, weather, start, end) in the song's time."""
    steps = TOUR
    if args.tour:
        steps = []
        for part in args.tour.split(","):
            fields = part.split(":")
            weather = fields[1] if len(fields) > 1 and fields[1] else "evening"
            steps.append((fields[0], weather, float(fields[2]) if len(fields) > 2 else 1.0))
    song = played(args)
    x, rate = read(args.work / "music" / f"{song['name']}.wav")
    end = len(x) / rate
    tour, t = [], 0.0
    for k, (listener, weather, phrases) in enumerate(steps):
        last = k == len(steps) - 1 or phrases <= 0
        phrases = args.last_phrases if last else phrases
        stop = end if phrases <= 0 else min(end, t + phrases * phrase_s(song))
        if stop > t:
            tour.append((listener, weather, t, stop))
        t = stop
    return tour


def gains_of(seats: dict, tour) -> dict[str, float]:
    """The gain (dB) each stop plays at: the loudest place's loudness everywhere, but a
    place in still air at the gain of the same place in the evening's weather. Loudness
    as meters weigh it (K), not the A weighting of the levels: the far places and the
    back of the stage keep little but bass, which the A weighting all but leaves out, so
    matching levels would turn their bass up far past the audience's."""

    def loudness(seat: dict) -> float:
        return seat.get("loudness_db", seat["rms_db"])

    top = max(loudness(v) for v in seats["seats"].values())
    gains = {}
    for listener, weather, _, _ in tour:
        key = f"{listener}.{weather}"
        evening = seats["seats"].get(f"{listener}.evening", seats["seats"][key])
        gains[key] = top - loudness(evening)
    return gains


def tour_audio(args, tour, seats) -> np.ndarray:
    rate = seats["rate"]
    length = round(tour[-1][3] * rate)
    out = np.zeros((length, 2))
    half = int(CROSSFADE_S * rate / 2)
    gains = gains_of(seats, tour)
    for k, (listener, weather, start, stop) in enumerate(tour):
        key = f"{listener}.{weather}"
        x = read(args.work / "seats" / f"{key}.{seats['format']}.wav")[0]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x = shifted(x, round(seats["seats"][key]["lag_s"] * rate), length)
        x = x * 10.0 ** (gains[key] / 20.0)
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
    import lakeside_pictures as pictures

    seats = json.loads((args.work / "seats" / "seats.json").read_text(encoding="utf-8"))
    tour = tour_of(args)
    missing = [f"{n}.{w}" for n, w, _, _ in tour if f"{n}.{w}" not in seats["seats"]]
    if missing:
        raise SystemExit(f"error: no mix for {missing[0]}; run: python lakeside_demo.py render")
    audio = tour_audio(args, tour, seats)
    out = args.work
    write(out / "tour.wav", audio, seats["rate"])
    print(f"wrote {out / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    drawing = pictures.Pictures(
        {w: yaml.safe_load(s.read_text(encoding="utf-8")) for w, (s, _) in weathers(args).items()},
        seats["seats"],
        tour,
        TITLES,
        gains_of(seats, tour),
        played(args)["credit"],
        folder=args.scene.parent,
    )
    stills = out / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, (listener, weather, start, stop) in enumerate(tour):
        path = stills / f"{k:02d}_{listener}_{weather}.png"
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
    parser.add_argument("step", choices=["music", "scenes", "render", "video", "all"])
    parser.add_argument(
        "--from", dest="source", type=Path, default=Path("scene.yaml"), help="the example's scene"
    )
    parser.add_argument(
        "--scene", type=Path, default=Path("scene_demo.yaml"), help="the scene the demo plays"
    )
    parser.add_argument(
        "--still", type=Path, default=Path("scene_still.yaml"), help="the same in still air"
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results_demo"), help="simulate's folder for it"
    )
    parser.add_argument(
        "--still-results",
        type=Path,
        default=Path("results_still"),
        help="simulate's folder for the scene in still air",
    )
    parser.add_argument(
        "--work", type=Path, default=Path("demo"), help="folder for everything this writes"
    )
    parser.add_argument(
        "--song",
        choices=sorted(SONGS),
        default="spicy_funk_cake",
        help="music: the song to play (default spicy_funk_cake)",
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
    parser.add_argument(
        "--tour",
        help="video: listener:weather:phrases,... (weather evening or still; 8 bars a phrase)",
    )
    parser.add_argument(
        "--last-phrases",
        type=float,
        default=2.0,
        help="video: phrases the last stop lasts (0: to the end of the song; default 2)",
    )
    parser.add_argument(
        "--no-mp4", action="store_true", help="video: write the audio and stills only"
    )
    args = parser.parse_args()
    if args.hrtf and args.format != "binaural":
        parser.error("--hrtf is for binaural mixes")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    steps = ["music", "render", "video"] if args.step == "all" else [args.step]
    for step in steps:
        globals()[step](args)


if __name__ == "__main__":
    main()
