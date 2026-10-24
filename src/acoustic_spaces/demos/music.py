"""The recordings the demos play: where each song's files are, how they are fetched and
summed, and the phrases the tours move on.

A song names its master and, where it has them, its stems, each as a list of files to
add together; a file is ``(url, member)``, one member of the zip archive at that URL
(read by HTTP range requests, so that one stem comes out of a release of several
gigabytes), or ``(url, None)``, the whole file. :func:`write_tracks` fetches what is
missing, decodes it (ffmpeg for anything but WAV) and writes one WAV per track, keeping
the files it downloaded so that a second run fetches nothing.

The tours move on the song's phrases: eight bars, from its first downbeat
(:func:`phrase_tour`), so that a cut falls where the music turns.
"""

from __future__ import annotations

import urllib.parse
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from acoustic_spaces import fetch
from acoustic_spaces.demos.audio import decoded, mono, resampled, stereo, write

#: Bars in a phrase: the tours move on every eighth bar.
PHRASE_BARS = 8
#: A song's file: its URL, and the member to take out of the zip archive there.
File = tuple[str, str | None]
Key = TypeVar("Key")


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
    tracks: Mapping[str, Sequence[File]],
    folder: Path,
    *,
    rate: int | None = None,
    channels: int | None = None,
) -> dict[str, Path]:
    """Each track of ``tracks`` (name to its files) as ``folder/<name>.wav``, at ``rate``
    and in ``channels`` channels (1 or 2) when given; tracks already there are kept."""
    if channels not in (None, 1, 2):
        raise ValueError(f"channels: 1 or 2, not {channels}")
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / f"{name}.wav" for name in tracks}
    with Archives() as archives:
        for name, files in tracks.items():
            if paths[name].exists():
                continue
            x, got = summed(files, folder, archives)
            if rate is not None:
                x, got = resampled(x, got, rate), rate
            if channels == 1:
                x = mono(x)
            elif channels == 2:
                x = stereo(x)
            write(paths[name], x, got)
            print(f"wrote {paths[name]} from {len(files)} file(s)")
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


__all__ = ["PHRASE_BARS", "Archives", "File", "phrase_s", "phrase_tour", "summed", "write_tracks"]
