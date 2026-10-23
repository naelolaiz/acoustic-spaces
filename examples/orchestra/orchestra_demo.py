"""Beethoven's 8th symphony in the concert hall example, heard from its four seats, and a
video that tours them through three movements.

The recording is the TU Berlin's anechoic one (Böhm, Ackermann and Weinzierl, Orchester
Wiener Akademie; CC BY 4.0, https://doi.org/10.14279/depositonce-6729.2): 61 tracks, one
per player, which the `orchestra` example puts on the stage as 61 sources. The three
movements it has (I, II and IV) come as movement1.zip, movement2.zip and movement4.zip
(560 to 710 MB each), downloaded by hand from the page behind that DOI. Run this from
the folder of the example (`acoustic-spaces examples --copy . orchestra && cd orchestra`),
in the Python environment acoustic-spaces is installed in, with ffmpeg on the PATH and
the zips in it (or `--zips FOLDER`):

    acoustic-spaces simulate scene.yaml --out results
    python orchestra_demo.py tracks   # the 61 tracks of each movement, out of the zips
    python orchestra_demo.py render   # each stop of the tour: a movement at a seat, binaural
    python orchestra_demo.py video    # demo/tour.wav, the stills, demo/tour.mp4

or `python orchestra_demo.py all` once the simulation is done. Everything goes to ./demo
(--work). Use headphones; `--hrtf me.sofa` on every step and on `simulate` plays through
your own or measured HRTFs instead of a spherical head.

The tour plays what the recording has of the three movements (70 to 85 s of each), each
from two to four of the seats in turn:
the first from the middle of the stalls, the front rows and the conductor's podium, the
second from the back of the hall and the middle, the finale from the podium, the front,
the back and the middle again. Within a movement the seat changes at the quietest moment
near each third (or half, or quarter) of it, so the cuts fall between phrases, with a 1 s
crossfade; between movements the hall's reverberation dies away and there is a pause
(the video runs about four minutes).
Every seat's mix is shifted earlier by the time the orchestra's sound takes to reach it
(10 ms on the podium, 85 ms at the back), so the music runs on across a cut. All stops
keep one gain (the loudest peak at -1 dBFS): the hall's differences are heard as the
simulation gives them, from the podium, 2 m from the first violins, to the back row,
and the captions give each stop's level against the loudest. Every source has the same
power, so the balance between the instruments is the recording's own.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

#: The movements the recording has: numeral and title, and the zip each comes in.
MOVEMENTS = {
    1: ("I", "Allegro vivace e con brio", "movement1"),
    2: ("II", "Allegretto scherzando", "movement2"),
    4: ("IV", "Allegro vivace", "movement4"),
}
#: Source name = track file, as in auralize.sh.
TRACKS = {
    "fl1": "01_Fl_1.wav",
    "fl2": "02_Fl_2.wav",
    "ob1": "03_Ob_1.wav",
    "ob2": "04_Ob_2.wav",
    "cl1": "05_Cl_1.wav",
    "cl2": "06_Cl_2.wav",
    "bn1": "07_Bn_1.wav",
    "bn2": "08_Bn_2.wav",
    "hn1": "09_Hrn_1.wav",
    "hn2": "10_Hrn_2.wav",
    "tp1": "11_Trp_1.wav",
    "tp2": "12_Trp_2.wav",
    "timpani": "13_Tp.wav",
    **{
        f"vl1_{k:02d}": f"{13 + k}_Vl_1_{1 + (k - 1) // 7}_{1 + (k - 1) % 7}.wav"
        for k in range(1, 15)
    },
    **{
        f"vl2_{k:02d}": f"{27 + k}_Vl_2_{1 + (k - 1) // 6}_{1 + (k - 1) % 6}.wav"
        for k in range(1, 7)
    },
    **{f"vl2_{k:02d}": f"{27 + k}_Vl_2_2_{k - 6}.wav" for k in range(7, 12)},
    **{f"vla_{k:02d}": f"{38 + k}_Vla_1_{k}.wav" for k in range(1, 7)},
    **{f"vla_{k:02d}": f"{38 + k}_Vla_2_{k - 6}.wav" for k in range(7, 12)},
    **{f"vc_{k}": f"{49 + k}_Vc_{k}.wav" for k in range(1, 7)},
    **{f"db_{k}": f"{55 + k}_Db_{k}.wav" for k in range(1, 7)},
}
#: The default tour: the movements in order, each with the seats that hear it in turn,
#: for equal shares of it (the cuts moved to the quietest moment near each share's end).
TOUR = [
    (1, ["stalls-middle", "stalls-front", "conductor"]),
    (2, ["stalls-rear", "stalls-middle"]),
    (4, ["conductor", "stalls-front", "stalls-rear", "stalls-middle"]),
]
TITLES = {
    "conductor": ("The conductor's podium", "on the stage, 2 m from the first violins"),
    "stalls-front": ("Front of the stalls", "4 m from the stage's edge"),
    "stalls-middle": ("Middle of the stalls", "12 m from the stage, left of the aisle"),
    "stalls-rear": ("Back of the stalls", "the last rows, 21 m from the stage"),
}
SHORT = {
    "conductor": "podium",
    "stalls-front": "front",
    "stalls-middle": "middle",
    "stalls-rear": "back",
}
#: How far (s) a cut may move from its share's end to a quieter moment.
QUIET_S = 10.0
CROSSFADE_S = 1.0
#: The pause between movements (s), after the hall has died away.
PAUSE_S = 3.0
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


def envelope(x: np.ndarray, rate: int, step_s: float = 0.1) -> np.ndarray:
    """RMS of ``x`` over 0.5 s windows every ``step_s``."""
    mono = x if x.ndim == 1 else x.mean(axis=1)
    step, window = int(step_s * rate), int(0.5 * rate)
    power = np.cumsum(np.r_[0.0, mono.astype(float) ** 2])
    starts = np.arange(0, max(1, len(mono) - window), step)
    return np.sqrt((power[starts + window] - power[starts]) / window)


def quiet_moment(env: np.ndarray, step_s: float, near_s: float, within_s: float) -> float:
    """The time (s) of the quietest moment within ``within_s`` of ``near_s``."""
    lo = max(0, int((near_s - within_s) / step_s))
    hi = min(len(env) - 1, int((near_s + within_s) / step_s))
    if hi <= lo:
        return near_s
    return (lo + int(np.argmin(env[lo : hi + 1]))) * step_s + 0.25


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


def movements_of(args) -> list[int]:
    return (
        [m for m, _ in TOUR] if not args.movements else [int(m) for m in args.movements.split(",")]
    )


def tracks_folder(args, movement: int) -> Path:
    return args.work / "tracks" / MOVEMENTS[movement][2]


def tracks(args) -> None:
    """The 61 tracks of each movement of the tour, out of movementN.zip (``--zips``, the
    example's folder by default), into demo/tracks/movementN; a folder already full is
    kept, as is one the zip was unpacked into by hand (movementN/anechoic)."""
    for movement in movements_of(args):
        name = MOVEMENTS[movement][2]
        folder = tracks_folder(args, movement)
        if all((folder / track).exists() for track in TRACKS.values()):
            print(f"{folder}: all {len(TRACKS)} tracks there")
            continue
        unpacked = args.zips / name / "anechoic"
        if all((unpacked / track).exists() for track in TRACKS.values()):
            print(f"{name}: tracks in {unpacked}")
            folder.mkdir(parents=True, exist_ok=True)
            for track in TRACKS.values():
                shutil.copyfile(unpacked / track, folder / track)
            continue
        zip_path = args.zips / f"{name}.zip"
        if not zip_path.exists():
            raise SystemExit(
                f"error: no {zip_path}: download it from https://doi.org/10.14279/depositonce-6729.2"
            )
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            members = {Path(info.filename).name: info for info in archive.infolist()}
            missing = [track for track in TRACKS.values() if track not in members]
            if missing:
                raise SystemExit(f"error: {zip_path} has no {missing[0]}")
            for track in TRACKS.values():
                with archive.open(members[track]) as source, (folder / track).open("wb") as target:
                    shutil.copyfileobj(source, target, 1 << 20)
        print(f"wrote {len(TRACKS)} tracks to {folder}")


def simulate_command(args) -> str:
    """The simulation the demo mixes from (the scene writes mono and binaural responses)
    and, with ``--hrtf``, the SOFA file the binaural ones go through."""
    return f"acoustic-spaces simulate {args.scene} --out {args.results}" + hrtf_option(args)


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
        raise SystemExit(
            f"error: no simulation in {args.results}; run first:\n  {simulate_command(args)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))["responses"]


def arrival_s(args, listener: str) -> float:
    """When the orchestra's sound reaches ``listener`` (s): the peak, over 2 ms, of the
    energy of its responses added up."""
    total, rate = np.zeros(0), 1
    for entry in responses_of(args):
        if entry["listener"] != listener or entry["source"] not in TRACKS:
            continue
        outputs = entry.get("outputs") or [{"wav": entry["wav"]}]
        h, rate = read(args.results / outputs[0]["wav"])
        energy = (h**2).sum(axis=1) if h.ndim > 1 else h**2
        total = np.pad(total, (0, max(0, len(energy) - len(total))))
        total[: len(energy)] += energy
    smooth = np.convolve(total, np.ones(max(1, int(0.002 * rate))), "same")
    return int(np.argmax(smooth)) / rate


def stops_of(args) -> list[tuple[int, str]]:
    """(movement, seat) for each stop of the tour."""
    if args.tour:
        stops = []
        for part in args.tour.split(","):
            movement, seat = part.split(":")
            stops.append((int(movement), seat))
        return stops
    return [(movement, seat) for movement, seats in TOUR for seat in seats]


def mix_path(args, movement: int, seat: str) -> Path:
    return args.work / "seats" / f"{MOVEMENTS[movement][2]}_{seat}.{args.format}.wav"


def render(args) -> None:
    """Each stop's mix: a movement at a seat (``--format``, binaural by default), then
    one gain for all of them (the loudest peak at -1 dBFS)."""
    from acoustic_spaces.cli import main as cli

    check_hrtf(args, responses_of(args))
    stops = stops_of(args)
    peaks, rate = {}, 0
    for movement, seat in stops:
        folder = tracks_folder(args, movement)
        if not all((folder / track).exists() for track in TRACKS.values()):
            raise SystemExit(f"error: no tracks in {folder}; run: python orchestra_demo.py tracks")
        out = mix_path(args, movement, seat)
        if out.exists() and not args.again:
            print(f"{out}: there already")
        else:
            pairs = [f"{name}={folder / track}" for name, track in TRACKS.items()]
            command = ["auralize", str(args.results), *pairs, "--listener", seat]
            code = cli(command + ["--format", output_label(args), "--out", str(out)])
            if code:
                raise SystemExit(code)
        x, rate = read(out)
        peaks[movement, seat] = float(np.abs(x).max())
    gain = 10.0 ** (-1.0 / 20.0) / max(peaks.values())
    info: dict[str, dict] = {}
    lags = {seat: arrival_s(args, seat) for seat in {seat for _, seat in stops}}
    for movement, seat in stops:
        x, rate = read(mix_path(args, movement, seat))
        level = gated_rms(a_weighted(x * gain, rate), rate)
        info[f"{movement}:{seat}"] = {
            "movement": movement,
            "seat": seat,
            "lag_s": lags[seat],
            "rms_db": 20.0 * math.log10(level + 1e-12),
            "length_s": len(x) / rate,
        }
    top = max(v["rms_db"] for v in info.values())
    for key, v in info.items():
        v["relative_db"] = v["rms_db"] - top
        late = v["lag_s"] * 1e3
        print(f"{key:>18}: {v['relative_db']:+6.1f} dB(A), sound arrives {late:4.0f} ms late")
    (args.work / "seats" / "seats.json").write_text(
        json.dumps({"rate": rate, "format": args.format, "gain": gain, "stops": info}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def tour_of(args, seats: dict) -> list[dict]:
    """The stops with their times: in each movement the seats take equal shares, each
    cut moved to the quietest moment within QUIET_S of its share's end (by the first
    seat's envelope); the movements follow each other with a pause between them."""
    stops = stops_of(args)
    tour, t = [], 0.0
    for movement in dict.fromkeys(m for m, _ in stops):
        here = [seat for m, seat in stops if m == movement]
        first = seats["stops"][f"{movement}:{here[0]}"]
        length = first["length_s"]
        env = envelope(read(mix_path(args, movement, here[0]))[0], seats["rate"])
        cuts = [0.0]
        for k in range(1, len(here)):
            cuts.append(quiet_moment(env, 0.1, length * k / len(here), QUIET_S))
        cuts.append(length)
        for seat, start, stop in zip(here, cuts[:-1], cuts[1:], strict=True):
            tour.append(
                {
                    "movement": movement,
                    "seat": seat,
                    "start": start,  # in the movement's time
                    "stop": stop,
                    "at": t + start,  # in the tour's time
                }
            )
        t += length + PAUSE_S
    return tour


def tour_audio(args, tour, seats) -> np.ndarray:
    rate = seats["rate"]
    last = tour[-1]
    length = round((last["at"] + last["stop"] - last["start"]) * rate)
    out = np.zeros((length, 2))
    half = int(CROSSFADE_S * rate / 2)
    for k, stop in enumerate(tour):
        x = read(mix_path(args, stop["movement"], stop["seat"]))[0] * seats["gain"]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        lag = round(seats["stops"][f"{stop['movement']}:{stop['seat']}"]["lag_s"] * rate)
        offset = round((stop["at"] - stop["start"]) * rate)  # the movement's start in the tour
        x = shifted(x, lag - offset, length)
        a, b = round(stop["at"] * rate), round((stop["at"] + stop["stop"] - stop["start"]) * rate)
        previous = k and tour[k - 1]["movement"] == stop["movement"]
        following = k < len(tour) - 1 and tour[k + 1]["movement"] == stop["movement"]
        w = np.zeros(length)
        w[(a - half if previous else a) : (b + half if following else b)] = 1.0
        ramp = np.sin(np.linspace(0.0, np.pi / 2, 2 * half))  # equal power
        if previous:
            w[a - half : a + half] = ramp
        if following:
            w[b - half : b + half] = ramp[::-1]
        out += x[:, :2] * w[:, None]
    fade = int(FADE_OUT_S * rate)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None] ** 2
    return out


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def video(args) -> None:
    """tour.wav, one still per stop of the tour, and tour.mp4 with a scrolling
    spectrogram under the stills (ffmpeg)."""
    import make_images as pictures

    seats = json.loads((args.work / "seats" / "seats.json").read_text(encoding="utf-8"))
    missing = [f"{m}:{s}" for m, s in stops_of(args) if f"{m}:{s}" not in seats["stops"]]
    if missing:
        raise SystemExit(f"error: no mix for {missing[0]}; run: python orchestra_demo.py render")
    tour = tour_of(args, seats)
    audio = tour_audio(args, tour, seats)
    out = args.work
    write(out / "tour.wav", audio, seats["rate"])
    print(f"wrote {out / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    scene = yaml.safe_load(args.scene.read_text(encoding="utf-8"))
    stills = out / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    lines, chapters = [], []
    for k, stop in enumerate(tour):
        seat, movement = stop["seat"], stop["movement"]
        facts = facts_of(seats["stops"][f"{movement}:{seat}"])
        drawing = pictures.Pictures(scene, seat, movement, facts=facts)
        img = drawing.cover()
        tour_bar(img, tour, k, pictures)
        path = stills / f"{k:02d}_{MOVEMENTS[movement][2]}_{seat}.png"
        img.convert("RGB").save(path, optimize=True)
        duration = stop["stop"] - stop["start"] + (PAUSE_S if not_last_of_movement(tour, k) else 0)
        lines += [f"file '{path.resolve().as_posix()}'", f"duration {duration:.3f}"]
        chapters.append(f"{mmss(stop['at'])} {MOVEMENTS[movement][0]} · {TITLES[seat][0]}")
    lines.append(lines[-2])  # the concat demuxer needs the last picture again
    (stills / "stills.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = tour[0]
    thumbnail = pictures.Pictures(scene, first["seat"], first["movement"]).thumbnail(
        note=f"3 movements · 4 seats · {len(TRACKS)} anechoic tracks"
    )
    thumbnail.convert("RGB").save(out / "thumbnail.png", optimize=True)
    (out / "chapters.txt").write_text("\n".join(chapters) + "\n", encoding="utf-8")
    print(
        f"wrote {len(tour)} stills to {stills}, {out / 'thumbnail.png'} and {out / 'chapters.txt'}"
    )
    if args.no_mp4:
        return
    if shutil.which("ffmpeg") is None:
        raise SystemExit("error: ffmpeg is needed for the video")
    spectrum = (
        "[1:a]showspectrum=s=1720x512:slide=scroll:color=magma:scale=log:fscale=log:"
        "start=40:stop=16000:drange=80:overlap=0.5:fps=30,scale=1720:96,"
        "format=rgba,colorkey=0x000000:0.08:0.25[s];"
        "[0:v]fps=30,format=rgba[b];[b][s]overlay=100:892:shortest=1,format=yuv420p[v]"
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


def not_last_of_movement(tour, k: int) -> bool:
    """Whether stop ``k`` ends its movement, so that the pause after it is its."""
    return k == len(tour) - 1 or tour[k + 1]["movement"] != tour[k]["movement"]


def facts_of(stop: dict) -> list[str]:
    relative = stop["relative_db"]
    return [
        "The loudest stop of the tour"
        if relative > -0.05
        else f"{relative:.1f} dB(A) against the loudest stop",
        f"The orchestra reaches you {stop['lag_s'] * 1e3:.0f} ms after it plays",
    ]


def tour_bar(img, tour, k: int, pictures) -> None:
    """A bar across the picture with a segment per stop, the current one bright, the
    movements' numerals over it."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    left, right, top, height = 80, 1840, 172, 22
    last = tour[-1]
    total = last["at"] + last["stop"] - last["start"]
    fonts = [pictures.font("SemiBold", size) for size in (17, 14)]
    for j, stop in enumerate(tour):
        x0 = left + (right - left) * stop["at"] / total
        x1 = left + (right - left) * (stop["at"] + stop["stop"] - stop["start"]) / total
        colour = pictures.LISTENER if j == k else (pictures.LINE if j < k else pictures.PANEL)
        draw.rounded_rectangle([x0, top, x1 - 3, top + height], 5, colour)
        label = f"{MOVEMENTS[stop['movement']][0]} · {SHORT[stop['seat']]}"
        # The label in the larger font that fits the segment, if one does.
        fits = [font for font in fonts if draw.textlength(label, font=font) < x1 - x0 - 8]
        if fits:
            fill = pictures.BG if j == k else pictures.MUTED
            draw.text(
                ((x0 + x1) / 2, top + height / 2), label, font=fits[0], fill=fill, anchor="mm"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["tracks", "render", "video", "all"])
    parser.add_argument(
        "--scene", type=Path, default=Path("scene.yaml"), help="the example's scene"
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results"), help="folder written by simulate"
    )
    parser.add_argument(
        "--zips", type=Path, default=Path("."), help="tracks: the folder with movementN.zip"
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
    parser.add_argument("--movements", help="tracks: only these movements, comma separated")
    parser.add_argument(
        "--tour",
        help="movement:seat,... (the seats of a movement take equal shares of it)",
    )
    parser.add_argument("--again", action="store_true", help="render: mix stops that exist")
    parser.add_argument(
        "--no-mp4", action="store_true", help="video: write the audio and stills only"
    )
    args = parser.parse_args()
    if args.hrtf and args.format != "binaural":
        parser.error("--hrtf is for binaural mixes")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    steps = ["tracks", "render", "video"] if args.step == "all" else [args.step]
    for step in steps:
        globals()[step](args)


if __name__ == "__main__":
    main()
