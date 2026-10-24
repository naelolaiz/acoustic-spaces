"""The recordings the demos play: where each song's files are, how they are fetched and
summed, and the phrases the tours move on.

A song names its master and, where it has them, its stems, each as a list of files to
add together; a file is ``(url, member)``, one member of the zip archive at that URL
(read by HTTP range requests, so that one stem comes out of a release of several
gigabytes), or ``(url, None)``, the whole file. :func:`write_tracks` fetches what is
missing, decodes it (ffmpeg for anything but WAV) and writes one WAV per track, keeping
the files it downloaded so that a second run fetches nothing. A track may instead be a
file of your own (a :class:`~pathlib.Path`), which is decoded and written the same way,
nothing downloaded: :func:`own_song_options` are the options a demo takes for a song of
yours in place of its presets, :func:`own_song_problems` what is wrong with them.

The tours move on the song's phrases: eight bars, from its first downbeat
(:func:`phrase_tour`), so that a cut falls where the music turns.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.parse
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from acoustic_spaces import fetch
from acoustic_spaces.demos import Option
from acoustic_spaces.demos.audio import decoded, mono, resampled, stereo, write

#: Bars in a phrase: the tours move on every eighth bar.
PHRASE_BARS = 8
#: A song's file: its URL, and the member to take out of the zip archive there.
File = tuple[str, str | None]
#: A track: the files to fetch and add together, or a file of your own.
Track = Sequence[File] | Path
Key = TypeVar("Key")
#: The editor's file filter for a song of your own: WAV, or what ffmpeg decodes.
AUDIO_FILTER = "Audio files (*.wav *.flac *.aiff *.aif *.mp3 *.m4a *.ogg *.opus);;All files (*)"


def phrase_s(song: Mapping[str, Any]) -> float:
    """A phrase of ``song`` (:data:`PHRASE_BARS` bars of four beats), s."""
    return PHRASE_BARS * 4 * 60.0 / song["bpm"]


class Archives:
    """The zip archives a download reads members from, each opened once and closed with
    the block."""

    def __init__(self) -> None:
        self.open: dict[str, zipfile.ZipFile] = {}

    def __enter__(self) -> Archives:
        return self

    def __exit__(self, *exception) -> None:
        for archive in self.open.values():
            archive.close()
        self.open.clear()

    def archive(self, url: str) -> zipfile.ZipFile:
        if url not in self.open:
            print(f"reading from {url}")
            self.open[url] = fetch.open_zip(url)
        return self.open[url]

    def file(self, url: str, member: str | None, folder: Path) -> Path:
        """``member`` of the archive at ``url``, or the whole file, in ``folder`` under
        its own name; one already there is kept."""
        name = Path(member or urllib.parse.urlparse(url).path).name
        path = folder / urllib.parse.unquote(name)
        if path.exists():
            return path
        if member:
            fetch.copy_member(self.archive(url), member, path)
        else:
            print(f"downloading {url}")
            fetch.download(url, path)
        return path


def summed(files: Sequence[File], folder: Path, archives: Archives) -> tuple[np.ndarray, int]:
    """The files fetched, decoded and added together (the shortest gives the length)."""
    total, rate = None, 0
    for url, member in files:
        x, rate = decoded(archives.file(url, member, folder))
        total = x if total is None else total[: len(x)] + x[: len(total)]
    if total is None:
        raise ValueError("a track with no files")
    return total, rate


def write_tracks(
    tracks: Mapping[str, Track],
    folder: Path,
    *,
    rate: int | None = None,
    channels: int | None = None,
) -> dict[str, Path]:
    """Each track of ``tracks`` (name to its files, or to a file of your own) as
    ``folder/<name>.wav``, at ``rate`` and in ``channels`` channels (1 or 2) when given.
    Tracks fetched before are kept; a file of your own is read again, as it may have
    changed."""
    if channels not in (None, 1, 2):
        raise ValueError(f"channels: 1 or 2, not {channels}")
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / f"{name}.wav" for name in tracks}
    with Archives() as archives:
        for name, files in tracks.items():
            if isinstance(files, Path):
                x, got = decoded(files)
                source = str(files)
            elif paths[name].exists():
                continue
            else:
                x, got = summed(files, folder, archives)
                source = f"{len(files)} file(s)"
            if rate is not None:
                x, got = resampled(x, got, rate), rate
            if channels == 1:
                x = mono(x)
            elif channels == 2:
                x = stereo(x)
            write(paths[name], x, got)
            print(f"wrote {paths[name]} from {source}")
    return paths


def phrase_tour(
    steps: Sequence[tuple[Key, float]],
    length_s: float,
    phrase: float,
    *,
    first_s: float = 0.0,
    last_phrases: float | None = None,
) -> list[tuple[Key, float, float]]:
    """Each stop of a tour as (key, start, stop) in the music's time, from ``steps``
    (key, phrases): the stops end on the phrase boundaries after ``first_s``, so that
    none drifts off the beat, and none runs past ``length_s``. A stop of no phrases runs
    to the end, as does the last one unless ``last_phrases`` says how long it lasts
    (``last_phrases`` 0 runs it to the end as well); a stop with no time left is left
    out."""
    tour: list[tuple[Key, float, float]] = []
    start, phrases = 0.0, 0.0
    for k, (key, count) in enumerate(steps):
        last = k == len(steps) - 1
        if last and last_phrases is not None:
            count = last_phrases
        if count <= 0 or (last and last_phrases is None):
            stop = length_s
        else:
            phrases += count
            stop = min(length_s, first_s + phrases * phrase)
        if stop > start:
            tour.append((key, start, stop))
        start = stop
    return tour


# --------------------------------------------------------------------------- your own song


def own_song_options(
    steps: tuple[str, ...] = ("music",),
    *,
    stems: Sequence[str] = (),
    downbeat: bool = False,
) -> list[Option]:
    """The options a demo takes for a song of your own in place of its presets, for
    ``steps``: ``--music FILE``, its master (stereo; WAV, or anything ffmpeg decodes),
    ``--bpm``, its tempo, which the tour's phrases of eight bars go by, ``--credit``, the
    line the video shows for it; where the demo plays stems, ``--stem NAME=WAV`` for
    each of ``stems``; and where its phrases run from a downbeat, ``--downbeat``, the
    time of the first one."""
    options = [
        Option(
            "music",
            "a song of your own in place of --song: its master, stereo (WAV, or anything "
            "ffmpeg decodes); nothing is downloaded, and the credit names the file",
            None,
            "file",
            steps=steps,
            filter=AUDIO_FILTER,
        ),
        Option(
            "bpm",
            "with --music: the song's tempo, beats a minute; the tour moves every 8 bars of it",
            None,
            "float",
            steps=steps,
        ),
    ]
    if downbeat:
        options.append(
            Option(
                "downbeat",
                "with --music: when the song's first downbeat falls, s, which its 8-bar "
                "phrases run from",
                0.0,
                "float",
                steps=steps,
            )
        )
    if stems:
        options.append(
            Option(
                "stems",
                f"with --music: a stem of the song, as STEM=WAV, once per stem "
                f"({', '.join(stems)}), for what plays stems; one not given is left out of "
                "its mixes, and a mix with none plays nothing",
                None,
                "files",
                tuple(stems),
                steps,
                flag="--stem",
                filter="WAV files (*.wav)",
            )
        )
    options.append(
        Option(
            "credit",
            "with --music: the credit line the video shows (the file's name unless given)",
            None,
            steps=steps,
        )
    )
    return options


def own_song_problems(demo: Any, args: argparse.Namespace) -> str | None:
    """What is wrong with a song of your own (:func:`own_song_options`): its other options
    without ``--music``, no ``--bpm`` or one of no tempo, a master that is not there, or
    one ffmpeg would decode without ffmpeg on the PATH."""
    music = getattr(args, "music", None)
    if music is None:
        names = {option.name for option in demo.options}
        given = [
            demo.option(name).option
            for name in ("bpm", "downbeat", "stems", "credit")
            if name in names and getattr(args, name, None) not in (None, 0.0, {})
        ]
        if given:
            return f"{', '.join(given)}: for a song of your own, with --music"
        return None
    bpm = getattr(args, "bpm", None)
    if bpm is None:
        return "--music needs --bpm, the song's tempo: the tour moves every 8 bars of it"
    if not bpm > 0:
        return f"--bpm: a tempo above 0, not {bpm}"
    path = Path(demo.resolved(args).music)
    if not path.is_file():
        return f"--music: no {path}"
    if path.suffix.lower() != ".wav" and shutil.which("ffmpeg") is None:
        return f"--music: ffmpeg is needed to decode {path.name}, and is not on the PATH"
    return None


__all__ = [
    "AUDIO_FILTER",
    "PHRASE_BARS",
    "Archives",
    "File",
    "Track",
    "own_song_options",
    "own_song_problems",
    "phrase_s",
    "phrase_tour",
    "summed",
    "write_tracks",
]
