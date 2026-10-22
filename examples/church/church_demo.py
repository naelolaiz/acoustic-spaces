"""A choir and an organ in the church example, heard at its listeners, and a video that
tours them.

Nineteen singers stand in two rows in front of the altar and sing the round "Dona nobis
pacem"; the organ in the gallery above the entrance plays it first, alone, and then
accompanies them. The singers are real: each one's dry take (a recording booth, the
microphone close) from VocalSet (CC BY 4.0), brought to one tempo and, partly, to one
tuning. The organ plays a four-part harmonisation of the tune, written for this demo,
from the pipe organ samples of the Versilian Community Sample Library (CC0). Run it from
the folder of the church example (`acoustic-spaces examples --copy . church && cd
church`), in the Python environment acoustic-spaces is installed in, with ffmpeg on the
PATH:

    python church_demo.py music     # download the 19 takes, their notes and the organ samples
    python church_demo.py scene     # scene_choir.yaml: the church with the choir and the organ
    acoustic-spaces simulate scene_choir.yaml --out results_choir --output binaural
    python church_demo.py feeds     # one WAV per singer and per side of the organ
    python church_demo.py render    # every listener, binaural
    python church_demo.py video     # demo/tour.wav, the stills, demo/tour.mp4

or `python church_demo.py all` once the simulation is done. Everything goes to ./demo
(--work). Use headphones.

The recordings are not part of acoustic_spaces: `music` downloads them from where their
authors publish them (about 100 MB; only the needed files are read out of VocalSet's and
Annotated-VocalSet's archives, by HTTP range requests) and writes their credits to
demo/music/SOURCES.txt. The singers are CC BY 4.0, so whoever shares a render must credit
VocalSet as that file says; the organ samples are CC0.

The music, at a crotchet a second: verse 1, the organ alone (an open diapason 8' and the
same rank an octave up as a 4', the pedal with its 16'); verse 2, the choir alone; verse
3, the choir with the organ on its quiet stops, slowing down into a held last chord and
the church's reverberation. The women sing the tune at its pitch, the men an octave lower,
as the takes were sung.

The impulse responses `simulate` writes are per unit of sound power, and `auralize`
scales each source by its level_db. So every feed is brought to one gated RMS where it
plays loudest (the organ's first verse, each singer's singing), and the scene gives the
sound powers: 88 dB re 1 pW for each singer (a choir singing mezzo forte), 105 dB for
each side of the organ's manual and 102 dB for each pedal tower; the quiet verse of the
organ sits 11 dB below its full one inside its feeds.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml

# --------------------------------------------------------------------------- sources

VOCALSET = "https://zenodo.org/api/records/1442513/files/VocalSet11.zip/content"
VOCALSET_DOI = "10.5281/zenodo.1442513"
ANNOTATIONS = "https://zenodo.org/api/records/7061507/files/Annotated%20VocalSet.zip/content"
ANNOTATIONS_DOI = "10.5281/zenodo.7061507"
VCSL_COMMIT = "c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e"
VCSL = f"https://raw.githubusercontent.com/sgossner/VCSL/{VCSL_COMMIT}/Aerophones/Edge-blown%20Aerophones/Pipe%20Organ/"

#: The singers of VocalSet (its voice types) and where each stands: the women in the
#: front row, the men on a step behind them, both facing the nave. The back row leaves
#: the middle free for the listener "in_the_choir". The countertenor (m9) is left out:
#: the annotation of that take belongs to a longer take than the one in the archive.
FRONT = ["f1", "f5", "f2", "f3", "f8", "f4", "f6", "f7", "f9"]
BACK = ["m1", "m4", "m2", "m10", "m3", "m5", "m8", "m7", "m6", "m11"]
VOICES = {
    "f1": "soprano",
    "f2": "soprano",
    "f3": "soprano",
    "f4": "soprano",
    "f5": "mezzo-soprano",
    "f6": "soprano",
    "f7": "soprano",
    "f8": "mezzo-soprano",
    "f9": "soprano",
    "m1": "baritone",
    "m2": "tenor",
    "m3": "tenor",
    "m4": "bass",
    "m5": "baritone",
    "m6": "baritone",
    "m7": "tenor",
    "m8": "bass",
    "m10": "bass-baritone",
    "m11": "tenor",
}
SINGERS = FRONT + BACK

# --------------------------------------------------------------------------- the music

#: The tune as VocalSet's singers sang it (F major, 3/4): MIDI note and beats. The
#: women sing it here, the men an octave lower.
TUNE = [
    (65, 0.5), (60, 0.5), (69, 2.0),
    (67, 0.5), (60, 0.5), (70, 2.0),
    (69, 1.0), (67, 1.0), (65, 1.0),
    (65, 1.0), (64, 1.0), (None, 1.0),
    (74, 1.0), (72, 0.5), (70, 0.5), (69, 0.5), (67, 0.5),
    (72, 1.5), (70, 0.5), (69, 1.0),
    (69, 0.5), (67, 0.5), (65, 1.0), (64, 1.0),
    (65, 2.0), (None, 1.0),
]  # fmt: skip
#: The organ: soprano (the tune, held over the singers' breath), alto and tenor on the
#: manual, bass on the pedal, in MIDI notes at 8' pitch; "~" ties into the next note.
#: | F | C7/E | F C/E Dm | Csus4 C | Bb F/A Gm7 | F Dm | Gm Csus4 C7 | F |
ORGAN = {
    "S": [
        (65, 0.5), (60, 0.5), (69, 2.0), (67, 0.5), (60, 0.5), (70, 2.0), (69, 1.0), (67, 1.0),
        (65, 1.0), (65, 1.0), (64, 2.0), (74, 1.0), (72, 0.5), (70, 0.5), (69, 0.5), (67, 0.5),
        (72, 1.5), (70, 0.5), (69, 1.0), (69, 0.5), (67, 0.5), (65, 1.0), (64, 1.0), (65, 3.0),
    ],
    "A": [
        (60, 3.0, "~"), (60, 3.0), (65, 1.0), (64, 1.0), (62, 1.0), (60, 3.0),
        (65, 3.0, "~"), (65, 3.0), (62, 1.0), (60, 1.0), (58, 1.0), (60, 3.0),
    ],
    "T": [
        (57, 3.0), (55, 3.0), (60, 2.0), (57, 1.0), (55, 3.0), (62, 1.0), (60, 1.0), (58, 1.0),
        (57, 3.0), (55, 3.0), (57, 3.0),
    ],
    "B": [
        (41, 3.0), (40, 3.0), (41, 1.0), (40, 1.0), (38, 1.0), (36, 3.0), (46, 1.0), (45, 1.0),
        (43, 1.0), (41, 2.0), (38, 1.0), (43, 1.0), (36, 2.0), (41, 3.0),
    ],
}  # fmt: skip
BEATS = 24  # beats in a verse
#: When each verse starts (s); a crotchet lasts 1 s, but verse 3 slows down at its end.
VERSES = {1: 0.0, 2: 26.0, 3: 52.0}
#: Verse 3: the length of each beat from beat 18 on (bar 7), and the last chord (s).
RITARDANDO = {18: 1.06, 19: 1.12, 20: 1.2}
LAST_CHORD_S = 5.0
RING_S = 8.0  # the church's reverberation after the last chord
#: Organ registrations: the manual's samples, a 4' (the same rank an octave up) and its
#: gain in dB, the pedal's samples; the quiet one's level against the full one's.
FULL = {"manual": "Loud", "pedal": "Loud Pedal", "four_foot_db": -6.0}
QUIET = {"manual": "Quiet", "pedal": "Quiet Pedal", "four_foot_db": None}
QUIET_DB = -11.0

#: Sound power (dB re 1 pW) of each singer, each side of the manual, each pedal tower.
SINGER_DB = 88.0
MANUAL_DB = 105.0
PEDAL_DB = 102.0
#: Voices are near omni in the bass and beam forwards in the treble.
VOICE_PATTERN = [
    "omni",
    "omni",
    "omni",
    "subcardioid",
    "subcardioid",
    "cardioid",
    "cardioid",
    "cardioid",
]
ORGAN_PATTERN = [
    "omni",
    "omni",
    "omni",
    "omni",
    "subcardioid",
    "subcardioid",
    "cardioid",
    "cardioid",
]

#: The organ in the gallery over the entrance: each division's pipes stand in two halves,
#: the C side (C, D, E, F#, G#, A#) to the south and the C# side to the north.
ORGAN_SOURCES = {
    "organ_manual_c": {"position": [1.2, -1.6, 9.5], "division": "manual", "side": 0},
    "organ_manual_cis": {"position": [1.2, 1.6, 9.5], "division": "manual", "side": 1},
    "organ_pedal_c": {"position": [0.9, -3.8, 8.6], "division": "pedal", "side": 0},
    "organ_pedal_cis": {"position": [0.9, 3.8, 8.6], "division": "pedal", "side": 1},
}
#: The listeners: name, position, yaw (0 faces the altar, +x).
LISTENERS = {
    "nave_back": ([5.0, 2.0, 1.2], 0.0),
    "side_aisle": ([13.0, -6.0, 1.2], 0.0),
    "nave_middle": ([18.0, -4.0, 1.2], 0.0),
    "nave_front": ([26.5, 2.5, 1.2], 0.0),
    "conductor": ([29.4, 0.0, 1.7], 0.0),
    "in_the_choir": ([31.8, 0.0, 1.95], 180.0),
}
#: The tour: listener and from when (s); each runs until the next one starts.
TOUR = [
    ("nave_back", 0.0),
    ("side_aisle", 11.5),
    ("nave_front", 25.0),
    ("conductor", 37.5),
    ("in_the_choir", 51.0),
    ("nave_middle", 63.5),
]
TITLES = {
    "nave_back": ("Back of the nave", "under the organ gallery, by the entrance"),
    "side_aisle": ("Side of the nave", "against the south wall, halfway down"),
    "nave_middle": ("Middle of the nave", "in the pews, a little to the south"),
    "nave_front": ("Front pews", "five metres from the choir"),
    "conductor": ("The conductor", "standing in front of the choir"),
    "in_the_choir": ("In the choir", "in the back row, among the men, facing the nave"),
}
CROSSFADE_S = 1.0

CTX = ssl.create_default_context()

#: Where the recordings come from, and the credit that renders shared need (the singers
#: are CC BY 4.0), as ``music`` writes it next to them.
CREDITS = f"""\
Singers: VocalSet by Julia Wilkins, Prem Seetharaman, Alison Wahl, Bryan Pardo (ISMIR
2018), https://doi.org/{VOCALSET_DOI}, CC BY 4.0
Note annotations: Annotated-VocalSet by Behnam Faghih, Joseph Timoney (Applied Sciences,
2022), https://doi.org/{ANNOTATIONS_DOI}, CC BY 4.0
Organ: Versilian Community Sample Library, Pipe Organ, sampled by Simon Dalzell (Ivy
Audio), Versilian Studios LLC, https://github.com/sgossner/VCSL (commit {VCSL_COMMIT}),
CC0 1.0
"Dona nobis pacem": traditional round, public domain; the organ's harmonisation was written
for this demo (church_demo.py, ORGAN).

Credit for sharing a render (the video, say):

Singers: VocalSet by J. Wilkins, P. Seetharaman, A. Wahl, B. Pardo (CC BY 4.0,
doi:{VOCALSET_DOI}), with note annotations from Annotated-VocalSet by B. Faghih and
J. Timoney (CC BY 4.0, doi:{ANNOTATIONS_DOI}); retuned, time-aligned and placed in a
simulated church. Organ: VCSL pipe organ, sampled by Simon Dalzell (Ivy Audio), Versilian
Studios (CC0). License: https://creativecommons.org/licenses/by/4.0/
"""


# --------------------------------------------------------------------------- audio


def read(path: Path) -> tuple[np.ndarray, int]:
    from acoustic_spaces.audio import read_wav

    return read_wav(path)


def write(path: Path, data: np.ndarray, rate: int) -> None:
    from acoustic_spaces.audio import write_wav

    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, data, rate)


def mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def gated_rms(x: np.ndarray, rate: int) -> float:
    """RMS over the 400 ms blocks that are playing: louder than -70 dBFS and than 20 dB
    below the mean of those (as loudness meters gate, without the K weighting)."""
    m = mono(x)
    block = int(0.4 * rate)
    n = len(m) // block
    if n == 0:
        return float(np.sqrt(np.mean(m**2)))
    power = (m[: n * block].reshape(n, block) ** 2).mean(axis=1)
    on = power[power > 1e-7]
    if on.size == 0:
        return 0.0
    on = on[on > on.mean() * 0.01]
    return float(np.sqrt(on.mean()))


def a_weighted(x: np.ndarray, rate: int) -> np.ndarray:
    """``x`` (samples, channels) through the A weighting (IEC 61672), by FFT."""
    n = 1 << math.ceil(math.log2(len(x)))
    f2 = np.fft.rfftfreq(n, 1.0 / rate) ** 2
    gain = (12194.0**2 * f2**2) / (
        (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
    )
    gain *= 10.0 ** (2.0 / 20.0)
    rows = np.ascontiguousarray(np.atleast_2d(x.T))
    return np.fft.irfft(np.fft.rfft(rows, n) * gain, n)[:, : len(x)].T


def cubic(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``x`` read at the fractional sample positions ``t`` (Catmull-Rom)."""
    i = np.floor(t).astype(np.int64)
    f = t - i
    p = np.pad(x, (1, 3))
    a, b, c, d = p[i], p[i + 1], p[i + 2], p[i + 3]
    return b + 0.5 * f * (
        c - a + f * (2.0 * a - 5.0 * b + 4.0 * c - d + f * (3.0 * (b - c) + d - a))
    )


def wsola(y: np.ndarray, out_times: np.ndarray, src_times: np.ndarray, length: int) -> np.ndarray:
    """``y`` played along a new time line, its pitch kept: output sample ``out_times[k]``
    comes from source sample ``src_times[k]``, linearly in between (WSOLA: overlapping
    windows, each shifted by up to ``TOLERANCE`` to continue the last one's waveform)."""
    frame, hop, tol = 1536, 384, 384
    win = np.hanning(frame)
    pad = frame + tol
    yp = np.pad(y, (pad, pad + length + frame))
    out = np.zeros(length + frame)
    norm = np.zeros(length + frame)
    n_fft = 1 << math.ceil(math.log2(frame + 2 * tol + frame))
    previous = None
    for k in range(0, length, hop):
        want = int(round(np.interp(k + frame / 2, out_times, src_times) - frame / 2))
        want = min(max(want, -frame), len(y))
        if previous is None:
            best = want
        else:
            ref = yp[pad + previous + hop : pad + previous + hop + frame] * win
            seg = yp[pad + want - tol : pad + want + tol + frame]
            c = np.fft.irfft(np.fft.rfft(seg, n_fft) * np.conj(np.fft.rfft(ref, n_fft)), n_fft)[
                : 2 * tol + 1
            ]
            best = want - tol + int(np.argmax(c))
        out[k : k + frame] += yp[pad + best : pad + best + frame] * win
        norm[k : k + frame] += win
        previous = best
    return out[:length] / np.maximum(norm[:length], 0.25)


# --------------------------------------------------------------------------- downloads


class RangeFile(io.RawIOBase):
    """A remote file read by HTTP range requests, so that zipfile can take a few members
    out of a large archive without downloading it."""

    BLOCK = 4 << 20

    def __init__(self, url: str):
        self.url, self.pos, self.cache = url, 0, {}
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, context=CTX, timeout=120) as response:
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
            for attempt in range(5):
                try:
                    request = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
                    with urllib.request.urlopen(request, context=CTX, timeout=300) as response:
                        data = response.read()
                    if len(data) == b - a + 1:
                        break
                except OSError as error:
                    if attempt == 4:
                        raise
                    print(f"retrying ({error})")
            if len(self.cache) > 16:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = data
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, offset = divmod(self.pos, self.BLOCK)
            chunk = self._block(i)[offset : offset + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def take_name(singer: str) -> str:
    return f"{singer}_dona_straight"


def organ_needed() -> dict[str, list[str]]:
    """The organ samples the arrangement needs: for each note the nearest sample (they
    are a minor third apart, so at most a semitone away)."""
    manual = sorted(
        {n[0] for v in "SAT" for n in ORGAN[v]} | {n[0] + 12 for v in "SAT" for n in ORGAN[v]}
    )
    pedal = sorted({n[0] for n in ORGAN["B"]})
    files = {}
    for folder, prefix, notes in (
        ("Loud", "Rode_Man3Open_", manual),
        ("Quiet", "NT5_Man3Quiet_", [n for n in manual if n <= 74]),
        ("Loud Pedal", "Rode_Pedal_", pedal),
        ("Quiet Pedal", "NT5_PedalQuiet_", pedal),
    ):
        suffix = "_rr1" if folder.startswith("Quiet") else ""
        division = "pedal" if folder.endswith("Pedal") else "manual"
        files[folder] = sorted(
            {f"{prefix}{sample_name(nearest_sample(n, division))}{suffix}.wav" for n in notes}
        )
    return files


NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def sample_name(midi: int) -> str:
    """The VCSL name of a note: C3 is middle C (MIDI 60)."""
    return f"{NAMES[midi % 12]}{midi // 12 - 2}"


#: The notes VCSL sampled (MIDI): C, D#, F# and A, from C1 (36) up to C6 on the
#: manual and to F#3 on the pedal.
SAMPLED = {"manual": range(36, 85), "pedal": range(36, 67)}


def nearest_sample(midi: int, division: str) -> int:
    """The sampled note nearest ``midi``."""
    notes = [m for m in SAMPLED[division] if m % 12 in (0, 3, 6, 9)]
    return min(notes, key=lambda m: (abs(m - midi), m))


def music(args) -> None:
    """Download the 19 takes of "Dona nobis pacem" (straight tone) and their note
    annotations out of the two Zenodo archives, by range requests (36 MB of 2.5 GB), and
    the organ samples the arrangement needs from VCSL (pinned commit)."""
    folder = args.work / "music"
    takes = folder / "vocalset"
    notes = folder / "annotations"
    organ = folder / "organ"
    for path in (takes, notes, organ):
        path.mkdir(parents=True, exist_ok=True)
    missing = [s for s in SINGERS if not (takes / f"{take_name(s)}.wav").exists()]
    if missing:
        print(f"reading {VOCALSET} (range requests)")
        archive = zipfile.ZipFile(RangeFile(VOCALSET))
        names = {
            Path(i.filename).name: i
            for i in archive.infolist()
            if "excerpts/straight/" in i.filename
        }
        for singer in missing:
            info = names[f"{take_name(singer)}.wav"]
            (takes / f"{take_name(singer)}.wav").write_bytes(archive.read(info))
            print(f"  {info.filename}")
        for extra in ("readme-anon.txt", "DataSetVocalises.pdf"):
            (takes / extra).write_bytes(archive.read(extra))
    missing = [s for s in SINGERS if not (notes / f"{take_name(s)}.csv").exists()]
    if missing:
        print(f"reading {ANNOTATIONS} (range requests)")
        archive = zipfile.ZipFile(RangeFile(ANNOTATIONS))
        want = {
            f"Annotated VocalSet/extended 1/with file header/straight/{take_name(s)}.csv": s
            for s in missing
        }
        for info in archive.infolist():
            if info.filename in want:
                (notes / Path(info.filename).name).write_bytes(archive.read(info))
                print(f"  {info.filename}")
    for sub, files in organ_needed().items():
        for name in files:
            path = organ / sub / name
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            url = VCSL + urllib.parse.quote(f"{sub}/{name}")
            with urllib.request.urlopen(url, context=CTX, timeout=120) as response:
                path.write_bytes(response.read())
            print(f"  {sub}/{name}")
    (folder / "SOURCES.txt").write_text(CREDITS, encoding="utf-8")
    print(f"music in {folder}")


# --------------------------------------------------------------------------- the scene


def choir_positions() -> dict[str, list[float]]:
    """Two gently curved rows, 0.6 m apart, in front of the altar (x 35)."""
    places = {}
    for k, singer in enumerate(FRONT):
        y = -0.3 * (len(FRONT) - 1) + 0.6 * k
        places[singer] = [round(31.0 - 0.05 * y * y, 3), round(y, 2), 1.6]
    back_y = [-3.0, -2.4, -1.8, -1.2, -0.6, 0.6, 1.2, 1.8, 2.4, 3.0]
    for singer, y in zip(BACK, back_y, strict=True):
        places[singer] = [round(31.9 - 0.05 * y * y, 3), y, 1.95]
    return places


def scene(args) -> None:
    """``--scene`` (scene_choir.yaml) written from ``--from`` (the example's scene.yaml):
    the same church with the choir and the organ as sources, and the tour's listeners."""
    base = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    base["name"] = "Church with a choir and an organ"
    sources = [
        {
            "name": f"singer_{singer}",
            "position": place,
            "orientation": {"yaw": 180.0},
            "level_db": SINGER_DB,
            "directivity": {"pattern": list(VOICE_PATTERN)},
        }
        for singer, place in choir_positions().items()
    ]
    sources += [
        {
            "name": name,
            "position": spec["position"],
            "orientation": {"yaw": 0.0},
            "level_db": MANUAL_DB if spec["division"] == "manual" else PEDAL_DB,
            "directivity": {"pattern": list(ORGAN_PATTERN)},
        }
        for name, spec in ORGAN_SOURCES.items()
    ]
    base["sources"] = sources
    base["listeners"] = [
        {"name": name, "position": position, "orientation": {"yaw": yaw}}
        for name, (position, yaw) in LISTENERS.items()
    ]
    # The bass rings for up to ten seconds here: keep the responses long enough.
    base.setdefault("simulation", {})["ir_length_s"] = max(
        9.0, base["simulation"].get("ir_length_s", 0.0)
    )
    args.scene.write_text(
        "# The church example with a choir in front of the altar and the organ in the gallery,\n"
        "# written by church_demo.py.\n"
        + yaml.safe_dump(base, sort_keys=False, default_flow_style=None, width=100),
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


# --------------------------------------------------------------------------- the time line


def beat_time(verse: int, beat: float) -> float:
    """When ``beat`` (from 0, crotchets) of ``verse`` falls (s)."""
    t = VERSES[verse]
    whole = int(math.floor(beat))
    for b in range(whole):
        t += RITARDANDO.get(b, 1.0) if verse == 3 else 1.0
    return t + (beat - whole) * (RITARDANDO.get(whole, 1.0) if verse == 3 else 1.0)


def note_times(verse: int) -> list[tuple[float, float]]:
    """Start and end (s) of each sung note of the tune in ``verse``; the last one is held
    as long as the organ's last chord in verse 3."""
    times, beat = [], 0.0
    for k, (midi, beats) in enumerate(TUNE):
        if midi is not None:
            end = beat_time(verse, beat + beats)
            if k == len(TUNE) - 2:
                end = beat_time(verse, beat) + (LAST_CHORD_S if verse == 3 else 2.4)
            times.append((beat_time(verse, beat), end))
        beat += beats
    return times


def end_of_music() -> float:
    return beat_time(3, 21.0) + LAST_CHORD_S + RING_S


# --------------------------------------------------------------------------- the choir


def annotation(path: Path) -> list[dict]:
    """The sung notes of a take (Annotated-VocalSet): start, end, mean F0 and the
    transition before each."""
    lines = path.read_text(encoding="utf-8").splitlines()
    head = next(k for k, line in enumerate(lines) if line.startswith("Sequence"))
    rows = list(csv.reader(lines[head + 1 :]))
    notes, transition = [], None
    for row in rows:
        if len(row) < 11:
            continue
        kind = row[4].strip()
        start, end = float(row[1]), float(row[2])
        if kind == "Transition":
            transition = start
        elif kind == "Sound":
            notes.append({"start": start, "end": end, "f0": float(row[10]), "after": transition})
            transition = None
        else:
            transition = None
    return notes


def sung_midi(singer: str) -> list[int]:
    shift = -12 if singer.startswith("m") else 0
    return [midi + shift for midi, _ in TUNE if midi is not None]


def tuned_and_timed(
    singer: str, x: np.ndarray, rate: int, notes: list[dict]
) -> dict[int, np.ndarray]:
    """The take of ``singer`` in every verse it sings: pitch moved towards the tune's (all
    of the singer's own offset, 60 % of each note's own error, at most 50 cents of it)
    and time warped so that each note starts on its beat."""
    target = sung_midi(singer)
    cents = []
    for note, midi in zip(notes, target, strict=True):
        c = 1200.0 * math.log2(note["f0"] / (440.0 * 2.0 ** ((midi - 69) / 12)))
        cents.append(c - 1200.0 * round(c / 1200.0))  # sung an octave off: no matter
    offset = float(np.median(cents))
    shift = [-(offset + 0.6 * float(np.clip(c - offset, -50.0, 50.0))) for c in cents]
    # Pitch: a correction per note, ramped over the transitions between them.
    grid_t = [0.0]
    grid_c = [shift[0]]
    for note, s in zip(notes, shift, strict=True):
        grid_t += [
            note["after"] if note["after"] is not None else note["start"] - 0.02,
            note["end"],
        ]
        grid_c += [s, s]
    grid_t.append(len(x) / rate)
    grid_c.append(shift[-1])
    order = np.argsort(grid_t, kind="stable")
    grid_t, grid_c = np.asarray(grid_t)[order], np.asarray(grid_c)[order]
    tau = np.arange(0, len(x) + 1, 64, dtype=float)
    ratio = 2.0 ** (np.interp(tau / rate, grid_t, grid_c) / 1200.0)
    # Read the take faster where the pitch goes up: u is the tuned take's time.
    u = np.concatenate([[0.0], np.cumsum(np.diff(tau) / ratio[:-1])])
    tuned = cubic(x, np.interp(np.arange(int(u[-1])), u, tau))

    def at(seconds: float) -> float:  # take time -> tuned take's samples
        return float(np.interp(seconds * rate, tau, u))

    out = {}
    for verse in (2, 3):
        times = note_times(verse)
        lead = 0.35  # the consonant before the first note
        src, dst = [at(notes[0]["start"] - lead)], [(times[0][0] - lead) * rate]
        for note, (start, _end) in zip(notes, times, strict=True):
            onset = (
                note["start"] if note["after"] is None else 0.5 * (note["after"] + note["start"])
            )
            src.append(at(onset))
            dst.append(start * rate)
        # The breath in bar 4 and the last note's end.
        src.append(at(notes[10]["end"]))
        dst.append(times[10][1] * rate - 0.05 * rate)
        src.append(at(notes[-1]["end"]))
        dst.append(times[-1][1] * rate)
        src.append(at(notes[-1]["end"]) + 0.3 * rate)
        dst.append((times[-1][1] + 0.3) * rate)
        order = np.argsort(dst)
        src, dst = np.asarray(src)[order], np.asarray(dst)[order]
        src = np.maximum.accumulate(src)  # the take never runs backwards
        first = int(dst[0])
        length = int(dst[-1]) - first
        y = wsola(tuned, dst - first, src, length)
        fade = int(0.08 * rate)
        y[:fade] *= np.linspace(0.0, 1.0, fade)
        y[-fade:] *= np.linspace(1.0, 0.0, fade)
        out[verse] = (first, y)
    return out


def choir_tracks(args, rate_out: int, length: int) -> dict[str, np.ndarray]:
    from acoustic_spaces.render import resample

    folder = args.work / "music"
    tracks = {}
    for singer in SINGERS:
        x, rate = read(folder / "vocalset" / f"{take_name(singer)}.wav")
        x = mono(x)
        notes = annotation(folder / "annotations" / f"{take_name(singer)}.csv")
        if len(notes) != sum(1 for midi, _ in TUNE if midi is not None):
            raise SystemExit(f"error: {singer}'s annotation has {len(notes)} notes")
        track = np.zeros(int(length * rate / rate_out) + rate)
        for _verse, (first, y) in tuned_and_timed(singer, x, rate, notes).items():
            track[first : first + len(y)] += y[: len(track) - first]
        track = resample(track, rate, rate_out) if rate != rate_out else track
        tracks[f"singer_{singer}"] = track[:length] / max(gated_rms(track, rate_out), 1e-9)
        print(f"  {singer} ({VOICES[singer]})")
    return tracks


# --------------------------------------------------------------------------- the organ


def organ_events(voices: str) -> list[tuple[int, float, float]]:
    """(MIDI, start beat, end beat) of the organ's notes in ``voices``; a key held by two
    voices at once sounds once, and repeated notes are let go a moment before."""
    events = []
    for v in voices:
        beat = 0.0
        tie_from = None
        for note in ORGAN[v]:
            midi, beats = note[0], note[1]
            start = beat if tie_from is None else tie_from
            if len(note) > 2 and note[2] == "~":
                tie_from = start
            else:
                events.append((midi, start, beat + beats))
                tie_from = None
            beat += beats
    merged = []
    for midi in sorted({e[0] for e in events}):
        spans = sorted((s, e) for m, s, e in events if m == midi)
        current = list(spans[0])
        for s, e in spans[1:]:
            if s < current[1] - 1e-9:
                current[1] = max(current[1], e)
            else:
                merged.append((midi, current[0], current[1]))
                current = [s, e]
        merged.append((midi, current[0], current[1]))
    return merged


class Samples:
    """The organ's samples: each note from the nearest one, retuned by resampling,
    started at its attack and stopped with a short release. The samples of a set are
    evened out first: each is brought to a straight line through the levels of all of
    them against pitch, as the microphones stood nearer some pipes than others."""

    def __init__(self, folder: Path, rate: int):
        self.folder, self.rate, self.cache, self.trend = folder, rate, {}, {}

    @staticmethod
    def midi_of(path: Path) -> int:
        name = re.search(r"_([A-G]#?)(\d)(_rr1)?\.wav$", path.name)
        return NAMES.index(name.group(1)) + 12 * (int(name.group(2)) + 2)

    @staticmethod
    def level_db(x: np.ndarray, rate: int) -> float:
        held = x[int(1.0 * rate) : int(4.0 * rate)]
        return 10.0 * math.log10(float(np.mean(held**2)) + 1e-20)

    def evened(self, sub: str, path: Path, x: np.ndarray, rate: int) -> np.ndarray:
        if sub not in self.trend:
            notes, levels = [], []
            for other in sorted((self.folder / sub).glob("*.wav")):
                y, r = read(other)
                notes.append(self.midi_of(other))
                levels.append(self.level_db(mono(y), r))
            self.trend[sub] = (
                np.polyfit(notes, levels, 1) if len(notes) > 1 else np.array([0.0, levels[0]])
            )
        gain = np.polyval(self.trend[sub], self.midi_of(path)) - self.level_db(x, rate)
        return x * 10.0 ** (float(np.clip(gain, -12.0, 12.0)) / 20.0)

    def load(self, sub: str, midi: int) -> tuple[np.ndarray, int]:
        key = (sub, midi)
        if key not in self.cache:
            base = nearest_sample(midi, "pedal" if sub.endswith("Pedal") else "manual")
            path = next(f for f in (self.folder / sub).glob("*.wav") if self.midi_of(f) == base)
            x, rate = read(path)
            x = self.evened(sub, path, mono(x), rate)
            level = np.abs(x)
            onset = max(0, int(np.argmax(level > level.max() * 0.05)) - int(0.01 * rate))
            x = x[onset:]
            factor = 2.0 ** ((midi - base) / 12.0) * rate / self.rate
            n = int((len(x) - 4) / factor)
            self.cache[key] = cubic(x, np.arange(n) * factor)
        return self.cache[key]

    def play(
        self, out: np.ndarray, sub: str, midi: int, start: float, end: float, gain: float
    ) -> None:
        x = self.load(sub, midi)
        release = int(0.09 * self.rate)
        a = int(start * self.rate)
        n = min(len(x), int((end - start) * self.rate) + release)
        y = x[:n].copy()
        y[-release:] *= np.cos(np.linspace(0.0, np.pi / 2, release)) ** 2
        b = min(len(out), a + n)
        out[a:b] += gain * y[: b - a]


def organ_tracks(args, rate: int, length: int) -> dict[str, np.ndarray]:
    """The organ's four feeds (each side of the manual and of the pedal): verse 1 on the
    full registration, verse 3 on the quiet one 11 dB down, both brought so that the
    full verse of each division plays at one gated RMS."""
    samples = Samples(args.work / "music" / "organ", rate)
    tracks = {name: np.zeros(length) for name in ORGAN_SOURCES}
    full = {name: np.zeros(length) for name in ORGAN_SOURCES}
    gap = 0.06  # a repeated note is let go this long before it is played again
    for verse, registration in ((1, FULL), (3, QUIET)):
        for division, voices in (("manual", "SAT"), ("pedal", "B")):
            for midi, start, end in organ_events(voices):
                t0, t1 = beat_time(verse, start), beat_time(verse, end)
                if end >= BEATS - 1e-9:
                    t1 = beat_time(verse, 21.0) + (LAST_CHORD_S if verse == 3 else 3.0)
                t1 -= gap
                ranks = [(registration[division], midi, 1.0)]
                if division == "manual" and registration["four_foot_db"] is not None:
                    ranks.append(
                        (
                            registration["manual"],
                            midi + 12,
                            10.0 ** (registration["four_foot_db"] / 20.0),
                        )
                    )
                for sub, key, gain in ranks:
                    # A pipe stands on its note's side: the C side for C D E F# G# A#.
                    name = next(
                        n
                        for n, s in ORGAN_SOURCES.items()
                        if s["division"] == division and s["side"] == key % 2
                    )
                    samples.play((full if verse == 1 else tracks)[name], sub, key, t0, t1, gain)
    # One gain per division: its full verse at the reference RMS over both sides.
    for division in ("manual", "pedal"):
        names = [n for n, s in ORGAN_SOURCES.items() if s["division"] == division]
        full_rms = math.sqrt(sum(gated_rms(full[n], rate) ** 2 for n in names) / len(names))
        quiet_rms = math.sqrt(sum(gated_rms(tracks[n], rate) ** 2 for n in names) / len(names))
        for n in names:
            tracks[n] = (full[n] / full_rms) + tracks[n] / quiet_rms * 10.0 ** (QUIET_DB / 20.0)
    return tracks


# --------------------------------------------------------------------------- steps


def feeds(args) -> None:
    """One WAV per source: each singer's tuned and timed take, each side of the organ,
    at one gated RMS where it plays loudest (the scene's level_db does the rest)."""
    entry = responses_of(args)[0]
    outputs = entry.get("outputs") or [{"wav": entry["wav"]}]
    rate = read(args.results / outputs[0]["wav"])[1]
    length = int(end_of_music() * rate)
    print("the choir:")
    tracks = choir_tracks(args, rate, length)
    print("the organ")
    tracks |= organ_tracks(args, rate, length)
    out = args.work / "feeds"
    peak = max(float(np.abs(t).max()) for t in tracks.values())
    scale = 0.5 / peak
    for name, track in tracks.items():
        write(out / f"{name}.wav", track * scale, rate)
    (out / "feeds.json").write_text(
        json.dumps(
            {"rate": rate, "length_s": length / rate, "scale": scale, "sources": sorted(tracks)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(tracks)} feeds to {out}")


def render(args) -> None:
    """Every listener's mix (``--format``, binaural by default), with one gain for all:
    the loudest peak at -1 dBFS."""
    from acoustic_spaces.cli import main as cli

    table = json.loads((args.work / "feeds" / "feeds.json").read_text(encoding="utf-8"))
    rate = table["rate"]
    listeners = list(LISTENERS)
    if args.listeners:
        listeners = args.listeners.split(",")
    pairs = [f"{name}={args.work / 'feeds' / f'{name}.wav'}" for name in table["sources"]]
    seats = args.work / "seats"
    peaks = {}
    for listener in listeners:
        out = seats / f"{listener}.{args.format}.wav"
        code = cli(
            [
                "auralize",
                str(args.results),
                *pairs,
                "--listener",
                listener,
                "--format",
                args.format,
                "--out",
                str(out),
            ]
        )
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
        info[listener] = {"rms_db": 20.0 * math.log10(level + 1e-12)}
    top = max(v["rms_db"] for v in info.values())
    for listener, v in info.items():
        v["relative_db"] = v["rms_db"] - top
        print(f"{listener:>14}: {v['relative_db']:+6.1f} dB(A)")
    (seats / "seats.json").write_text(
        json.dumps({"rate": rate, "format": args.format, "gain": gain, "seats": info}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def tour_of(args) -> list[tuple[str, float, float]]:
    steps = TOUR
    if args.tour:
        steps = [(part.split(":")[0], float(part.split(":")[1])) for part in args.tour.split(",")]
    end = end_of_music()
    return [
        (name, t, steps[k + 1][1] if k + 1 < len(steps) else end)
        for k, (name, t) in enumerate(steps)
    ]


def tour_audio(args, tour, seats) -> np.ndarray:
    rate = seats["rate"]
    length = round(tour[-1][2] * rate)
    out = np.zeros((length, 2))
    half = int(CROSSFADE_S * rate / 2)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, 2 * half))  # equal power
    for k, (listener, start, stop) in enumerate(tour):
        x = read(args.work / "seats" / f"{listener}.{seats['format']}.wav")[0]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x = x[:length] if len(x) >= length else np.pad(x, ((0, length - len(x)), (0, 0)))
        a, b = round(start * rate), round(stop * rate)
        w = np.zeros(length)
        w[(a - half if k else 0) : (b + half if k < len(tour) - 1 else length)] = 1.0
        if k:
            w[a - half : a + half] = ramp
        if k < len(tour) - 1:
            w[b - half : b + half] = ramp[::-1]
        out += x[:, :2] * w[:, None]
    fade = int(2.0 * rate)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None] ** 2
    return out


def playing(t: float) -> tuple[str, ...]:
    """Who plays in the verse at ``t`` (s)."""
    if t < VERSES[2] - 1.0:
        return ("organ",)
    if t < VERSES[3] - 1.0:
        return ("choir",)
    return ("choir", "organ")


def what_plays(t: float) -> str:
    if t < VERSES[2] - 1.0:
        return "Verse 1: the organ alone, full"
    if t < VERSES[3] - 1.0:
        return "Verse 2: the choir alone"
    return "Verse 3: the choir with the organ, quiet"


def video(args) -> None:
    """tour.wav, one still per stop of the tour, and tour.mp4 with a scrolling
    spectrogram under the stills (ffmpeg)."""
    import church_pictures as pictures

    seats = json.loads((args.work / "seats" / "seats.json").read_text(encoding="utf-8"))
    tour = tour_of(args)
    missing = [name for name, _, _ in tour if name not in seats["seats"]]
    if missing:
        raise SystemExit(f"error: no mix for {missing[0]}; run: python church_demo.py render")
    audio = tour_audio(args, tour, seats)
    out = args.work
    write(out / "tour.wav", audio, seats["rate"])
    print(f"wrote {out / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    report = json.loads((args.results / "parameters.json").read_text(encoding="utf-8"))
    drawing = pictures.Pictures(
        scene_of(args), report, seats["seats"], tour, TITLES, what_plays, playing, VOICES
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
    parser.add_argument("step", choices=["music", "scene", "feeds", "render", "video", "all"])
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        default=Path("scene.yaml"),
        help="scene: the example's scene",
    )
    parser.add_argument(
        "--scene", type=Path, default=Path("scene_choir.yaml"), help="the scene with the choir"
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results_choir"), help="folder written by simulate"
    )
    parser.add_argument(
        "--work", type=Path, default=Path("demo"), help="folder for everything this writes"
    )
    parser.add_argument(
        "--format", default="binaural", help="output format the mixes take (default binaural)"
    )
    parser.add_argument("--listeners", help="render: only these listeners, comma separated")
    parser.add_argument("--tour", help="video: listener:start_s,... (each runs until the next)")
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
