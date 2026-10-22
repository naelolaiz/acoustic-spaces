"""A song played by a band on the stadium example's stage and through its sound system,
heard at each of its listeners, and a video that tours them.

The sound system (main arrays, subwoofers, fills, towers and delays) plays the stereo
master of "Rubidium" by Software-Entwicklungskit (album "Everything is Free", CC0 1.0,
https://github.com/ichbinsoftware/everythingisfree); the monitors play mixes of its
stems: the singer's wedges the lead vocal, the drum fill the drums, the side fills a
mix for the whole stage. Run it from the folder of the stadium example
(`acoustic-spaces examples --copy . stadium_concert && cd stadium_concert`), in the Python
environment acoustic-spaces is installed in, with ffmpeg on the PATH:

    python stadium_demo.py music     # download the master and stems (M4A), decode them
    python stadium_demo.py band      # scene_band.yaml: the scene with the band on the stage
    acoustic-spaces simulate scene_band.yaml --out results_band --output binaural
    python stadium_demo.py feeds     # one signal per loudspeaker, as loud as the scene says
    python stadium_demo.py render    # every listener, binaural
    python stadium_demo.py video     # demo/tour.wav, the stills, demo/tour.mp4

or `python stadium_demo.py all` once the simulation is done. Everything goes to
./demo (--work). Use headphones.

The impulse responses `simulate` writes carry each source's feed (gain_db, eq_db,
delay_ms) but are per unit of sound power. `feeds` scales each loudspeaker's signal by
its level_db against the loudest one's, after bringing every signal to the same gated
RMS, so the balance between the main arrays (134 dB), the fills (116 to 130 dB) and
the monitors is the scene's, and the feed files are what each loudspeaker plays.
`render` runs `auralize` with `--equal-power`, so the levels are applied once.

`video` plays the song once, moving between listeners on phrase boundaries (every 8
bars), with a 1 s crossfade. Each listener's mix is shifted earlier by the time the
sound system takes to reach them (0.1 s at the mix position, 0.4 s at the far end),
so that the beat carries on across the cut; the caption gives that time and the level
there against the loudest listener's. All listeners share one gain, so the level
differences between them are heard as the simulation gives them.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import yaml

ALBUM = "https://github.com/ichbinsoftware/everythingisfree"
COMMIT = "e1d0d5c0422d2626c211d5d877e6f69c37ce8712"
RAW = f"https://raw.githubusercontent.com/ichbinsoftware/everythingisfree/{COMMIT}/src/5.Rubidium/"
MASTER = "5.Rubidium_Master"
STEMS = [
    "5.Rubidium_Stem_BGVOX",
    "5.Rubidium_Stem_BUILD",
    "5.Rubidium_Stem_DRUMS 2",
    "5.Rubidium_Stem_DRUMS MAIN",
    "5.Rubidium_Stem_DRUMS OUTRO",
    "5.Rubidium_Stem_MELODY BREAK",
    "5.Rubidium_Stem_MINILOGUE_SYNTH",
    "5.Rubidium_Stem_PLUCK",
    "5.Rubidium_Stem_VOX LEAD",
]
BPM = 132.0
PHRASE_S = 8 * 4 * 60.0 / BPM  # 8 bars; the master's first downbeat is at its start

#: Monitor mixes: stem (without the "5.Rubidium_Stem_" prefix) and its gain in dB.
SINGER_MIX = {"VOX LEAD": 0, "BGVOX": -6, "MINILOGUE_SYNTH": -10, "PLUCK": -12, "DRUMS MAIN": -16}
DRUMMER_MIX = {
    "DRUMS MAIN": 0,
    "DRUMS 2": 0,
    "DRUMS OUTRO": 0,
    "BUILD": -8,
    "MINILOGUE_SYNTH": -6,
    "VOX LEAD": -6,
}
# The side fills carry the song for the whole stage, with the drums well down: the kit
# itself is loud on stage, so the players hear the drums from where the drummer is.
STAGE_MIX = (
    {stem.split("Stem_")[1]: 0 for stem in STEMS}
    | {"VOX LEAD": 3}
    | {drums: -20 for drums in ("DRUMS MAIN", "DRUMS 2", "DRUMS OUTRO", "BUILD")}
)
MONITORS = {"vocal_wedges": "singer", "drum_fill": "drummer", "side_fill": "stage"}
#: The band itself, as sources on the stage (added by ``band``): sound power (dB re 1 pW)
#: of a rock drum kit, a singer singing loud and an instrument amplifier turned up. The
#: song has no guitar, so the guitarist's amplifier plays its synthesizer parts.
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
BAND_MIX = {
    "drum_kit": {"DRUMS MAIN": 0, "DRUMS 2": 0, "DRUMS OUTRO": 0, "BUILD": 0},
    "singer_voice": {"VOX LEAD": 0},
    "guitar_amp": {"MINILOGUE_SYNTH": 0, "PLUCK": 0, "MELODY BREAK": 0},
}

#: The default tour: listener and how many phrases it lasts; the last one runs to the end.
TOUR = [
    ("mix_position", 2),
    ("pitch_front", 1),
    ("singer", 2),
    ("drummer", 1),
    ("guitarist", 1),
    ("pitch_side", 1),
    ("north_stand_near", 1),
    ("north_stand_far", 1),
    ("north_stand_top", 2),
    ("east_stand", 2),
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


def music(args) -> None:
    """Download the master and the stems (M4A) and decode them to WAV with ffmpeg.
    WAV files already in the folder (the uncompressed ones from the album's site, say)
    are kept."""
    folder = args.work / "music"
    folder.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("error: ffmpeg is needed to decode the M4A files")
    for name in [MASTER, *STEMS]:
        wav = folder / f"{name}.wav"
        if wav.exists():
            continue
        m4a = folder / f"{name}.m4a"
        if not m4a.exists():
            url = RAW + urllib.parse.quote(f"{name}.m4a")
            print(f"downloading {url}")
            with urllib.request.urlopen(url) as response:
                m4a.write_bytes(response.read())
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(m4a), "-c:a", "pcm_f32le", str(wav)],
            check=True,
        )
        print(f"wrote {wav}")
    (folder / "SOURCES.txt").write_text(
        'Software-Entwicklungskit, "Rubidium", from the album "Everything is Free"\n'
        f"{ALBUM} (commit {COMMIT}), src/5.Rubidium\n"
        "License: CC0 1.0 Universal (public domain dedication),\n"
        "https://creativecommons.org/publicdomain/zero/1.0/\n",
        encoding="utf-8",
    )


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
    return f"acoustic-spaces simulate {args.scene} --out {args.results} --output {args.format}"


def responses_of(args) -> list[dict]:
    path = args.results / "parameters.json"
    if not path.exists():
        raise SystemExit(
            f"error: no simulation in {args.results}; run first:\n  {simulate_command(args)}"
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
    folder = args.work / "music"
    if not (folder / f"{MASTER}.wav").exists():
        raise SystemExit("error: no music yet; run: python stadium_demo.py music")
    from acoustic_spaces.render import resample

    def load(name: str) -> np.ndarray:
        x, r = read(folder / f"{name}.wav")
        return resample(x, r, rate) if r != rate else x

    master = load(MASTER)
    stems = {name.split("Stem_")[1]: load(name) for name in STEMS}
    length = len(master)
    # The stems start a bar of silence before the master: line them up with it.
    together = sum(s.mean(axis=1)[: rate * 90] for s in stems.values())
    lag = lag_of(together, master.mean(axis=1)[: rate * 90], rate, 5.0)
    stems = {k: shifted(v, lag, length) for k, v in stems.items()}
    print(f"stems line up with the master {lag / rate:+.3f} s later")

    def mix(gains: dict[str, float]) -> np.ndarray:
        return sum(stems[k].mean(axis=1) * 10.0 ** (g / 20.0) for k, g in gains.items())

    signals = {
        "left": master[:, 0],
        "right": master[:, 1],
        "mono": master.mean(axis=1),
        "singer": mix(SINGER_MIX),
        "drummer": mix(DRUMMER_MIX),
        "stage": mix(STAGE_MIX),
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
    (out / "feeds.json").write_text(
        json.dumps({"rate": rate, "length_s": length / rate, "sources": table}, indent=2) + "\n",
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
    listeners = sorted({e["listener"] for e in responses_of(args)})
    if args.listeners:
        listeners = [name for name in args.listeners.split(",")]
    pairs = [f"{name}={args.work / 'feeds' / f'{name}.wav'}" for name in table["sources"]]
    seats = args.work / "seats"
    peaks = {}
    for listener in listeners:
        out = seats / f"{listener}.{args.format}.wav"
        # The feeds carry each source's level_db already: auralize is not to apply it.
        command = ["auralize", str(args.results), *pairs, "--listener", listener]
        code = cli(command + ["--format", args.format, "--out", str(out), "--equal-power"])
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
        lag = arrival_s(args, listener, table)
        info[listener] = {"lag_s": lag, "rms_db": 20.0 * math.log10(level + 1e-12)}
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
    default tour that have no mix (a scene without the stage's players, say)."""
    steps = TOUR
    if args.tour:
        steps = [
            (part.split(":")[0], float(part.split(":")[1]) if ":" in part else 1.0)
            for part in args.tour.split(",")
        ]
    else:
        skipped = sorted({name for name, _ in steps if name not in rendered})
        if skipped:
            print(f"note: no mix for {', '.join(skipped)}; the tour goes without them")
        steps = [(name, n) for name, n in steps if name in rendered]
    table = json.loads((args.work / "feeds" / "feeds.json").read_text(encoding="utf-8"))
    end = table["length_s"]
    tour, t = [], 0.0
    for k, (listener, phrases) in enumerate(steps):
        stop = end if k == len(steps) - 1 or phrases <= 0 else min(end, t + phrases * PHRASE_S)
        if stop > t:
            tour.append((listener, t, stop))
        t = stop
    return tour


def tour_audio(args, tour, seats) -> np.ndarray:
    rate = seats["rate"]
    length = round(tour[-1][2] * rate)
    out = np.zeros((length, 2))
    half = int(CROSSFADE_S * rate / 2)
    for k, (listener, start, stop) in enumerate(tour):
        x = read(args.work / "seats" / f"{listener}.{seats['format']}.wav")[0]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x = shifted(x, round(seats["seats"][listener]["lag_s"] * rate), length)
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
    return out


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
    drawing = pictures.Pictures(
        scene_of(args), args.scene.parent, levels, seats["seats"], tour, TITLES
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
    parser.add_argument("--listeners", help="render: only these listeners, comma separated")
    parser.add_argument(
        "--tour", help="video: listener:phrases,... (8 bars each; the last one runs to the end)"
    )
    parser.add_argument(
        "--no-mp4", action="store_true", help="video: write the audio and stills only"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    steps = ["music", "feeds", "render", "video"] if args.step == "all" else [args.step]
    for step in steps:
        globals()[step](args)


if __name__ == "__main__":
    main()
