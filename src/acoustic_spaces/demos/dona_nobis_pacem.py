"""The church's choir and organ, "Dona nobis pacem", built for the recordings
catalogue (:mod:`acoustic_spaces.recordings`, ``dona_nobis_pacem``): one stem per
singer and per side of the organ's divisions, which the church example's sources play.

Nineteen singers stand in two rows in front of the altar and sing the round "Dona nobis
pacem"; the organ in the gallery above the entrance plays it first, alone, and then
accompanies them. The singers are real: each one's dry take (a recording booth, the
microphone close) from VocalSet (CC BY 4.0), brought to one tempo and, partly, to one
tuning. The organ plays a four-part harmonisation of the tune, written for the church's
demo, from the pipe organ samples of the Versilian Community Sample Library (CC0).
:func:`build` downloads them from where their authors publish them (127 MB; only the
needed files are read out of VocalSet's and Annotated-VocalSet's archives, by HTTP range
requests), builds the stems and adds them up into the recording's master.

The music, at a crotchet a second: verse 1, the organ alone (an open diapason 8' and the
same rank an octave up as a 4', the pedal with its 16'); verse 2, the choir alone; verse
3, the choir with the organ on its quiet stops, slowing down into a held last chord and
the church's reverberation. The women sing the tune at its pitch, the men an octave lower,
as the takes were sung.

Each stem is built at one gated RMS where it plays loudest
(:data:`acoustic_spaces.recordings.LEVEL_RMS`: each singer's singing, each division's
first verse over its two sides), the quiet verse of the organ 11 dB below its full one,
so that a source playing a stem plays at its sound power there: the church's scene
gives 88 dB re 1 pW for each singer (a choir singing mezzo forte), 105 dB for each side
of the organ's manual and 102 dB for each pedal tower.
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import urllib.parse
from pathlib import Path
from typing import NamedTuple

import numpy as np

from acoustic_spaces import fetch, recordings
from acoustic_spaces.demos.audio import gated_rms, mono, read, resampled, write

# --------------------------------------------------------------------------- sources

VOCALSET = "https://zenodo.org/api/records/1442513/files/VocalSet11.zip/content"
VOCALSET_DOI = "10.5281/zenodo.1442513"
ANNOTATIONS = "https://zenodo.org/api/records/7061507/files/Annotated%20VocalSet.zip/content"
ANNOTATIONS_DOI = "10.5281/zenodo.7061507"
VCSL_COMMIT = "c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e"
VCSL = f"https://raw.githubusercontent.com/sgossner/VCSL/{VCSL_COMMIT}/Aerophones/Edge-blown%20Aerophones/Pipe%20Organ/"
#: Bytes a range request reads of the two archives: the takes are 1 to 2 MB each, far
#: apart in them.
BLOCK = 1 << 20
#: MB :func:`build` downloads: the takes and their notes out of the two archives, by range
#: requests (their 2.5 GB are not), and the organ samples.
MUSIC_MB = 127

#: The singers of VocalSet (its voice types), in the order the scene's Choir group stands
#: them: the women in the front row, the men on a step behind them, both facing the
#: nave. The countertenor (m9) is left out: the annotation of that take belongs to a
#: longer take than the one in the archive.
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
#: A note of the organ's: MIDI, beats, and "~" where it ties into the next.
OrganNote = tuple[int, float] | tuple[int, float, str]
ORGAN: dict[str, list[OrganNote]] = {
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


class Registration(NamedTuple):
    """The organ's stops in a verse: the sample sets of the manual and of the pedal, and
    the manual's 4' stop to its 8' (dB; None: off)."""

    manual: str
    pedal: str
    four_foot_db: float | None

    def of(self, division: str) -> str:
        return self.manual if division == "manual" else self.pedal


FULL = Registration("Loud", "Loud Pedal", -6.0)
QUIET = Registration("Quiet", "Quiet Pedal", None)
QUIET_DB = -11.0

#: The organ's stems, which the church's organ pipes play: each division's pipes stand
#: in two halves, the C side (C, D, E, F#, G#, A#) to the south and the C# side to the
#: north.
ORGAN_SOURCES = {
    "organ_manual_c": {"division": "manual", "side": 0},
    "organ_manual_cis": {"division": "manual", "side": 1},
    "organ_pedal_c": {"division": "pedal", "side": 0},
    "organ_pedal_cis": {"division": "pedal", "side": 1},
}
#: Every stem, named after the source of the church that plays it, in the order of its
#: groups: the singers (Choir), then the organ (Organ pipes).
SOURCE_NAMES = [f"singer_{singer}" for singer in SINGERS] + list(ORGAN_SOURCES)
#: The rate the stems are built at.
RATE = 48_000

# --------------------------------------------------------------------------- audio


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


def download(folder: Path) -> None:
    """The 19 takes of "Dona nobis pacem" (straight tone) and their note annotations out
    of the two Zenodo archives, by range requests (69 MB of 2.5 GB), and the organ
    samples the arrangement needs from VCSL (pinned commit), into ``folder``; those
    there already are kept, so that a download cut short goes on where it stopped."""
    takes = folder / "vocalset"
    notes = folder / "annotations"
    organ = folder / "organ"
    missing = [s for s in SINGERS if not (takes / f"{take_name(s)}.wav").exists()]
    if missing:
        print(f"reading {VOCALSET} (range requests)")
        with fetch.open_zip(VOCALSET, BLOCK) as archive:
            names = {
                Path(name).name: name for name in archive.namelist() if "excerpts/straight/" in name
            }
            for singer in missing:
                member = names[f"{take_name(singer)}.wav"]
                fetch.copy_member(archive, member, takes / f"{take_name(singer)}.wav")
                print(f"  {member}")
    missing = [s for s in SINGERS if not (notes / f"{take_name(s)}.csv").exists()]
    if missing:
        print(f"reading {ANNOTATIONS} (range requests)")
        with fetch.open_zip(ANNOTATIONS, BLOCK) as archive:
            for singer in missing:
                member = (
                    "Annotated VocalSet/extended 1/with file header/straight/"
                    f"{take_name(singer)}.csv"
                )
                fetch.copy_member(archive, member, notes / f"{take_name(singer)}.csv")
                print(f"  {member}")
    for sub, files in organ_needed().items():
        for name in files:
            path = organ / sub / name
            if not path.exists():
                fetch.download(VCSL + urllib.parse.quote(f"{sub}/{name}"), path)
                print(f"  {sub}/{name}")


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
) -> dict[int, tuple[int, np.ndarray]]:
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
    at_t, at_c = np.asarray(grid_t)[order], np.asarray(grid_c)[order]
    tau = np.arange(0, len(x) + 1, 64, dtype=float)
    ratio = 2.0 ** (np.interp(tau / rate, at_t, at_c) / 1200.0)
    # Read the take faster where the pitch goes up: u is the tuned take's time.
    u = np.concatenate([[0.0], np.cumsum(np.diff(tau) / ratio[:-1])])
    tuned = cubic(x, np.interp(np.arange(int(u[-1])), u, tau))

    def at(seconds: float) -> float:  # take time -> tuned take's samples
        return float(np.interp(seconds * rate, tau, u))

    out: dict[int, tuple[int, np.ndarray]] = {}
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
        read_at = np.maximum.accumulate(np.asarray(src)[order])  # never runs backwards
        put_at = np.asarray(dst)[order]
        first = int(put_at[0])
        length = int(put_at[-1]) - first
        y = wsola(tuned, put_at - first, read_at, length)
        fade = int(0.08 * rate)
        y[:fade] *= np.linspace(0.0, 1.0, fade)
        y[-fade:] *= np.linspace(1.0, 0.0, fade)
        out[verse] = (first, y)
    return out


def choir_tracks(folder: Path, rate_out: int, length: int) -> dict[str, np.ndarray]:
    """Each singer's take (in ``folder``, :func:`download`) tuned and timed, at
    ``rate_out`` and ``length`` samples long, at one gated RMS (LEVEL_RMS)."""
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
        track = resampled(track, rate, rate_out)[:length]
        level = recordings.LEVEL_RMS / max(gated_rms(track, rate_out), 1e-9)
        tracks[f"singer_{singer}"] = track * level
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
            if note[2:] == ("~",):
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
        self.folder, self.rate = folder, rate
        self.cache: dict[tuple[str, int], np.ndarray] = {}
        self.trend: dict[str, np.ndarray] = {}  # a line through the levels, dB per MIDI note

    @staticmethod
    def midi_of(path: Path) -> int:
        name = re.search(r"_([A-G]#?)(\d)(_rr1)?\.wav$", path.name)
        if name is None:
            raise ValueError(f"{path.name}: not a note's sample (C4.wav, F#2_rr1.wav)")
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

    def load(self, sub: str, midi: int) -> np.ndarray:
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


def organ_tracks(folder: Path, rate: int, length: int) -> dict[str, np.ndarray]:
    """The organ's four stems (each side of the manual and of the pedal), from the
    samples in ``folder`` (:func:`download`): verse 1 on the full registration, verse 3
    on the quiet one 11 dB down, both brought so that the full verse of each division
    plays at one gated RMS over its two sides (LEVEL_RMS)."""
    samples = Samples(folder / "organ", rate)
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
                ranks = [(registration.of(division), midi, 1.0)]
                if division == "manual" and registration.four_foot_db is not None:
                    ranks.append(
                        (registration.manual, midi + 12, 10.0 ** (registration.four_foot_db / 20.0))
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
            quiet = tracks[n] / quiet_rms * 10.0 ** (QUIET_DB / 20.0)
            tracks[n] = (full[n] / full_rms + quiet) * recordings.LEVEL_RMS
    return tracks


# --------------------------------------------------------------------------- the build


def build(item: recordings.Recording, folder: str | Path | None = None) -> None:
    """The recording's stems, one per singer and per side of the organ's divisions
    (:data:`SOURCE_NAMES`), into its stems' folder in ``folder``
    (:func:`~acoustic_spaces.recordings.stems_folder`), at RATE and as long as the music
    and the church's reverberation after it, and its master, the stems added up, from
    what :func:`download` fetches into a folder there, removed once they are built."""
    out = recordings.stems_folder(item, folder)
    sources = out / "sources"
    download(sources)
    length = int(end_of_music() * RATE)
    print("the choir:")
    tracks = choir_tracks(sources, RATE, length)
    print("the organ")
    tracks |= organ_tracks(sources, RATE, length)
    for name, track in tracks.items():
        path = recordings.stem_file(item, name, folder)
        part = path.with_name(path.name + ".part")
        write(part, track, RATE)
        part.replace(path)
    master = recordings.file_of(item, folder)
    part = master.with_name(master.name + ".part")
    total = np.zeros(length)
    for track in tracks.values():
        total[: len(track)] += track[:length]
    write(part, total, RATE)
    part.replace(master)
    print(f"wrote {len(tracks)} stems to {out} and their sum to {master}")
    shutil.rmtree(sources)


__all__ = ["SOURCE_NAMES", "build"]
