"""Downloads: whole files, with progress, and single members of remote archives.

A zip archive on a server that answers HTTP range requests (Zenodo, GitHub, most
repositories) is opened as a file read block by block (:class:`RangeFile`), so that
:mod:`zipfile` takes a few megabytes out of an archive of a gigabyte without the rest.
A tar archive has no directory to seek in: :func:`extract_tar` reads it as it comes
and keeps only the members asked for, stopping once it has them all. The demos fetch
their recordings this way and the editor its HRTF sets.
"""

from __future__ import annotations

import io
import os
import shutil
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path

from acoustic_spaces._version import __version__

#: Seconds to wait for a server, and how many times a block of a remote archive is asked
#: for before the read fails.
TIMEOUT = 120
RETRIES = 5
#: Bytes copied at a time, and reported to the progress callback.
CHUNK = 1 << 20
#: ``progress(done, total)``: bytes so far, and the total (None when the server does not
#: say). It may raise to stop the download; the part written so far is removed.
Progress = Callable[[int, int | None], None]

_CONTEXT = ssl.create_default_context()


def data_folder(name: str, variable: str) -> Path:
    """``acoustic-spaces/<name>`` under the user's data folder, where downloads are kept
    (``%LOCALAPPDATA%`` on Windows, ``~/Library/Application Support`` on macOS,
    ``$XDG_DATA_HOME`` or ``~/.local/share`` elsewhere); the environment variable
    ``variable`` names another folder instead."""
    named = os.environ.get(variable)
    if named:
        return Path(named)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "acoustic-spaces" / name


def urlopen(url: str, headers: dict[str, str] | None = None, timeout: float = TIMEOUT):
    request = urllib.request.Request(
        url, headers={"User-Agent": f"acoustic-spaces/{__version__}", **(headers or {})}
    )
    return urllib.request.urlopen(request, context=_CONTEXT, timeout=timeout)


class RangeFile(io.RawIOBase):
    """A remote file read by HTTP range requests, in blocks kept for reuse, so that
    zipfile can take a few members out of a large archive without downloading it.
    Each request costs a round trip to the server: larger blocks (``block``, bytes)
    read large members faster, smaller ones waste less around small ones."""

    BLOCK = 4 << 20
    #: Bytes of blocks kept (at least two blocks).
    KEPT = 64 << 20

    def __init__(self, url: str, block: int = BLOCK):
        self.url, self.pos, self.cache = url, 0, {}
        self.block, self.kept = block, max(2, self.KEPT // block)
        with urlopen(url, {"Range": "bytes=0-0"}) as response:
            content_range = response.headers.get("Content-Range")
            if content_range is None:
                raise OSError(f"{url}: the server does not answer range requests")
            self.size = int(content_range.split("/")[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            a, b = i * self.block, min(self.size, (i + 1) * self.block) - 1
            data = b""
            for attempt in range(RETRIES):
                try:
                    with urlopen(self.url, {"Range": f"bytes={a}-{b}"}, 300) as response:
                        data = response.read()
                    if len(data) == b - a + 1:
                        break
                except OSError:
                    if attempt == RETRIES - 1:
                        raise
            if len(data) != b - a + 1:  # cut short every time: never pass it on as whole
                raise OSError(f"{self.url}: bytes {a}-{b} came {len(data)} long, {RETRIES} times")
            if len(self.cache) >= self.kept:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = data
        return self.cache[i]

    def read(self, n: int | None = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, offset = divmod(self.pos, self.block)
            chunk = self._block(i)[offset : offset + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def open_zip(url: str, block: int = RangeFile.BLOCK) -> zipfile.ZipFile:
    """A remote zip archive, its members read on demand by range requests of ``block``
    bytes (:class:`RangeFile`)."""
    return zipfile.ZipFile(io.BufferedReader(RangeFile(url, block), 1 << 20))


def _copy(source, target, total: int | None, progress: Progress | None) -> None:
    done = 0
    while True:
        chunk = source.read(CHUNK)
        if not chunk:
            break
        target.write(chunk)
        done += len(chunk)
        if progress is not None:
            progress(done, total)


def download(url: str, path: str | Path, progress: Progress | None = None) -> Path:
    """Download ``url`` to ``path`` (its folders made), through a ``.part`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with urlopen(url) as response, part.open("wb") as target:
            length = response.headers.get("Content-Length")
            _copy(response, target, int(length) if length else None, progress)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)
    return path


def extract(url: str, member: str, path: str | Path, progress: Progress | None = None) -> Path:
    """Take ``member`` out of the zip archive at ``url`` into ``path``, reading only the
    archive's directory and that member."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with open_zip(url) as archive:
            info = archive.getinfo(member)
            with archive.open(info) as source, part.open("wb") as target:
                _copy(source, target, info.file_size, progress)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)
    return path


class _Counted(io.RawIOBase):
    """A response read through, its bytes counted for ``progress``."""

    def __init__(self, response, total: int | None, progress: Progress | None):
        self.response, self.total, self.progress, self.done = response, total, progress, 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        data = self.response.read(len(buffer))
        buffer[: len(data)] = data
        self.done += len(data)
        if self.progress is not None and data:
            self.progress(self.done, self.total)
        return len(data)


def extract_tar(
    url: str, members: Mapping[str, str | Path], progress: Progress | None = None
) -> dict[str, Path]:
    """Take ``members`` (names in the archive to paths) out of the tar archive at
    ``url``, compressed or not, reading it as it comes and stopping once every member
    is out. ``progress`` counts the bytes read of the archive."""
    wanted = {name: Path(path) for name, path in members.items()}
    done: dict[str, Path] = {}
    parts: list[Path] = []
    try:
        with urlopen(url) as response:
            length = response.headers.get("Content-Length")
            stream = io.BufferedReader(
                _Counted(response, int(length) if length else None, progress)
            )
            with tarfile.open(fileobj=stream, mode="r|*") as archive:
                for info in archive:
                    path = wanted.get(info.name)
                    if path is None or not info.isfile():
                        continue
                    source = archive.extractfile(info)
                    if source is None:
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    part = path.with_name(path.name + ".part")
                    parts.append(part)
                    with source, part.open("wb") as target:
                        shutil.copyfileobj(source, target, CHUNK)
                    part.replace(path)
                    done[info.name] = path
                    if len(done) == len(wanted):
                        break
    finally:
        for part in parts:
            part.unlink(missing_ok=True)
    missing = sorted(set(wanted) - set(done))
    if missing:
        raise OSError(f"{url}: no {missing[0]} in the archive")
    return done


def fetch(
    url: str, path: str | Path, member: str | None = None, progress: Progress | None = None
) -> Path:
    """:func:`download` the file at ``url``, or :func:`extract` ``member`` of the zip
    archive there."""
    if member:
        return extract(url, member, path, progress)
    return download(url, path, progress)


def copy_member(archive: zipfile.ZipFile, member: str, path: str | Path) -> Path:
    """Write ``member`` of an open archive to ``path``, in chunks, through a ``.part``
    file (so that a download cut short leaves no file that looks whole)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with archive.open(member) as source, part.open("wb") as target:
            shutil.copyfileobj(source, target, CHUNK)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)
    return path


__all__ = [
    "CHUNK",
    "RangeFile",
    "copy_member",
    "data_folder",
    "download",
    "extract",
    "extract_tar",
    "fetch",
    "open_zip",
    "urlopen",
]
