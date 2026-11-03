"""WAV files: 32-bit float output, and input in the common PCM and float formats."""

from __future__ import annotations

import contextlib
import io
import os
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

_PCM, _FLOAT, _EXTENSIBLE = 1, 3, 0xFFFE


#: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, the sub-format of extensible float WAV files.
_FLOAT_GUID = struct.pack("<IHH8B", 3, 0, 0x10, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71)
#: The largest size a RIFF file's 32-bit fields hold: a larger file is written as RF64
#: (EBU Tech 3306), whose ds64 chunk holds the sizes in 64 bits and whose RIFF and data
#: sizes say 0xFFFFFFFF.
_RIFF_LIMIT = 0xFFFFFFFF
_UNKNOWN = 0xFFFFFFFF
#: Bytes read or written at a time.
_PIECE = 1 << 22


def write_wav(
    path: str | Path, data: np.ndarray, sample_rate: int, channel_mask: int | None = None
) -> None:
    """Write samples (frames,) or (frames, channels) as 32-bit float WAV.

    Float WAV keeps levels above full scale, so relative levels between files survive.
    Files with more than two channels, or a ``channel_mask`` (WAVE_FORMAT_EXTENSIBLE
    speaker positions; 0 for none, as in AmbiX), use the extensible format. A file of
    more than 4 GiB is written as RF64 (EBU Tech 3306), the WAV format whose sizes take
    64 bits. The samples go out a few MiB at a time, so a long file takes no copy of
    itself in memory.
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("data must be (frames,) or (frames, channels)")
    frames, channels = data.shape
    if channels == 0:
        raise ValueError("data must have a channel at least")
    size = 4 * frames * channels
    block = 4 * channels
    extensible = channels > 2 or channel_mask is not None
    fmt = struct.pack(
        "<HHIIHH",
        _EXTENSIBLE if extensible else _FLOAT,
        channels,
        int(sample_rate),
        int(sample_rate) * block,
        block,
        32,
    )
    if extensible:
        fmt += struct.pack("<HHI", 22, 32, int(channel_mask or 0)) + _FLOAT_GUID
    else:
        fmt += struct.pack("<H", 0)
    fact = struct.pack("<I", min(frames, _UNKNOWN))
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"fact" + struct.pack("<I", len(fact))
    chunks += fact
    riff = 4 + len(chunks) + 8 + size
    if riff <= _RIFF_LIMIT:
        head = b"RIFF" + struct.pack("<I", riff) + b"WAVE" + chunks
        head += b"data" + struct.pack("<I", size)
    else:
        # ds64: the RIFF size, the data size and the frames (the fact chunk's), no table.
        ds64 = struct.pack("<QQQI", riff + 36, size, frames, 0)
        head = b"RF64" + struct.pack("<I", _UNKNOWN) + b"WAVE"
        head += b"ds64" + struct.pack("<I", len(ds64)) + ds64 + chunks
        head += b"data" + struct.pack("<I", _UNKNOWN)
    step = max(1, (1 << 20) // channels)  # frames of 4 MiB
    with open(path, "wb") as file:
        file.write(head)
        for start in range(0, frames, step):
            file.write(np.ascontiguousarray(data[start : start + step], dtype="<f4").tobytes())


def _chunks(file: BinaryIO, path: str | Path) -> list[tuple[bytes, int, int]]:
    """The chunks of a RIFF, RF64 or BW64 WAV file: (tag, offset of the body, size),
    sizes cut to what the file holds; in RF64 and BW64 (ITU-R BS.2088), sizes of
    0xFFFFFFFF are taken from the ds64 chunk."""
    end = file.seek(0, os.SEEK_END)
    file.seek(0)
    header = file.read(12)
    if len(header) < 12 or header[:4] not in (b"RIFF", b"RF64", b"BW64") or header[8:] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    wide: dict[bytes, int] = {}
    found = []
    pos = 12
    while pos + 8 <= end:
        file.seek(pos)
        tag, size = struct.unpack("<4sI", file.read(8))
        if tag == b"ds64" and header[:4] != b"RIFF":
            body = file.read(min(size, end - pos - 8))
            if len(body) < 28:
                raise ValueError(f"{path}: ds64 chunk too short")
            _, wide[b"data"], _, count = struct.unpack("<QQQI", body[:28])
            for at in range(28, min(28 + 12 * count, len(body) - 11), 12):
                name, length = struct.unpack("<4sQ", body[at : at + 12])
                wide[name] = length
        elif size == _UNKNOWN and tag in wide:
            size = wide[tag]
        found.append((tag, pos + 8, min(size, end - pos - 8)))
        pos += 8 + size + (size % 2)
    return found


def _sized(file: BinaryIO) -> bool:
    """Whether ``file`` seeks and ends where seeking to its end says: not a pipe, nor a
    file of procfs (size 0) or sysfs (4096), or a file growing as it is read. Leaves it
    at its start."""
    if not file.seekable():
        return False
    try:
        end = file.seek(0, os.SEEK_END)
        if end == 0:
            return False
        file.seek(end - 1)
        return len(file.read(2)) == 1
    except OSError:
        return False
    finally:
        file.seek(0)


@contextlib.contextmanager
def _wav_file(path: str | Path) -> Iterator[BinaryIO]:
    """``path`` opened to read, all of it read first where it does not seek to its end
    (see :func:`_sized`)."""
    with Path(path).open("rb") as file:
        yield file if _sized(file) else io.BytesIO(file.read())


def _stream_fmt(file: BinaryIO, path: str | Path) -> bytes | None:
    """The first 40 bytes of the fmt chunk of a WAV file read front to back, and no
    further than that chunk (a pipe may stay open), or None if it has none."""
    header = file.read(12)
    if len(header) < 12 or header[:4] not in (b"RIFF", b"RF64", b"BW64") or header[8:] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    wide: dict[bytes, int] = {}
    while len(chunk := file.read(8)) == 8:
        tag, size = struct.unpack("<4sI", chunk)
        used = 0
        if tag == b"ds64" and header[:4] != b"RIFF":
            body = file.read(min(size, 28))
            if len(body) < 28:
                raise ValueError(f"{path}: ds64 chunk too short")
            _, wide[b"data"], _, count = struct.unpack("<QQQI", body[:28])
            table = file.read(min(size - 28, 12 * count))
            for at in range(0, len(table) - 11, 12):
                name, length = struct.unpack("<4sQ", table[at : at + 12])
                wide[name] = length
            used = len(body) + len(table)
        elif size == _UNKNOWN and tag in wide:
            size = wide[tag]
        if tag == b"fmt ":
            return file.read(min(size, 40))
        left = size + size % 2 - used
        while left > 0 and (piece := file.read(min(left, _PIECE))):
            left -= len(piece)
    return None


def _samples(raw: bytes, kind: int, bits: int) -> np.ndarray:
    """Samples of ``bits`` bits of format ``kind`` as float64: PCM scaled to [-1, 1)."""
    if kind == _FLOAT:
        return np.frombuffer(raw, dtype=f"<f{bits // 8}").astype(np.float64)
    if bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)) | (
            b[:, 2].astype(np.int8).astype(np.int32) << 16
        )
        return ints / float(2**23)
    return np.frombuffer(raw, dtype=f"<i{bits // 8}") / float(2 ** (bits - 1))


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as float64 (frames, channels) and its sample rate.

    PCM 16, 24 and 32 bit are scaled to [-1, 1); float 32 and 64 bit are kept as they are.
    RF64 and BW64 files (WAV with 64-bit sizes, for files over 4 GiB) are read too. The
    samples come in a few MiB at a time, so the file takes no copy of itself in memory.
    """
    with _wav_file(path) as file:
        fmt, data = None, None
        for tag, offset, size in _chunks(file, path):
            if tag == b"fmt ":
                file.seek(offset)
                body = file.read(min(size, 40))
                fmt = struct.unpack("<HHIIHH", body[:16])
                if fmt[0] == _EXTENSIBLE and len(body) >= 26:
                    fmt = (struct.unpack("<H", body[24:26])[0], *fmt[1:])
            elif tag == b"data":
                data = offset, size
        if fmt is None or data is None:
            raise ValueError(f"{path}: missing fmt or data chunk")
        kind, channels, sample_rate, _, _, bits = fmt
        if (kind, bits) not in ((_FLOAT, 32), (_FLOAT, 64), (_PCM, 16), (_PCM, 24), (_PCM, 32)):
            raise ValueError(f"{path}: unsupported WAV format {kind} with {bits} bits")
        if channels == 0:
            raise ValueError(f"{path}: a WAV file of no channels")
        frame = bits // 8 * channels
        frames = data[1] // frame
        samples = np.empty((frames, channels))
        file.seek(data[0])
        step = max(1, _PIECE // frame)
        for start in range(0, frames, step):
            count = min(step, frames - start)
            samples[start : start + count] = _samples(file.read(count * frame), kind, bits).reshape(
                count, channels
            )
    return samples, int(sample_rate)


def channel_mask(path: str | Path) -> int | None:
    """The speaker positions (WAVE_FORMAT_EXTENSIBLE channel mask) of a WAV file, or
    ``None`` for a file in the plain format."""
    with Path(path).open("rb") as file:
        body = None
        if _sized(file):
            for tag, offset, size in _chunks(file, path):
                if tag == b"fmt ":
                    file.seek(offset)
                    body = file.read(min(size, 40))
                    break
        else:
            body = _stream_fmt(file, path)
    if body is None:
        raise ValueError(f"{path}: missing fmt chunk")
    kind = struct.unpack("<H", body[:2])[0]
    if kind == _EXTENSIBLE and len(body) >= 24:
        return int(struct.unpack("<I", body[20:24])[0])
    return None


__all__ = ["channel_mask", "read_wav", "write_wav"]
