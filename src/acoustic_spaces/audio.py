"""WAV files: 32-bit float output, and input in the common PCM and float formats."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_PCM, _FLOAT, _EXTENSIBLE = 1, 3, 0xFFFE


#: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, the sub-format of extensible float WAV files.
_FLOAT_GUID = struct.pack("<IHH8B", 3, 0, 0x10, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71)


def write_wav(
    path: str | Path, data: np.ndarray, sample_rate: int, channel_mask: int | None = None
) -> None:
    """Write samples (frames,) or (frames, channels) as 32-bit float WAV.

    Float WAV keeps levels above full scale, so relative levels between files survive.
    Files with more than two channels, or a ``channel_mask`` (WAVE_FORMAT_EXTENSIBLE
    speaker positions; 0 for none, as in AmbiX), use the extensible format.
    """
    data = np.asarray(data, dtype="<f4")
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("data must be (frames,) or (frames, channels)")
    channels = data.shape[1]
    payload = np.ascontiguousarray(data).tobytes()
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
    fact = struct.pack("<I", data.shape[0])
    chunks = [
        b"fmt " + struct.pack("<I", len(fmt)) + fmt,
        b"fact" + struct.pack("<I", len(fact)) + fact,
        b"data" + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) % 2 else b""),
    ]
    body = b"WAVE" + b"".join(chunks)
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as float64 (frames, channels) and its sample rate.

    PCM 16, 24 and 32 bit are scaled to [-1, 1); float 32 and 64 bit are kept as they are.
    """
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        tag, size = raw[pos : pos + 4], struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        body = raw[pos + 8 : pos + 8 + size]
        if tag == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            if fmt[0] == _EXTENSIBLE and len(body) >= 26:
                fmt = (struct.unpack("<H", body[24:26])[0], *fmt[1:])
        elif tag == b"data":
            data = body
        pos += 8 + size + (size % 2)
    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    kind, channels, sample_rate, _, _, bits = fmt
    if kind == _FLOAT and bits in (32, 64):
        samples = np.frombuffer(data, dtype=f"<f{bits // 8}").astype(np.float64)
    elif kind == _PCM and bits in (16, 32):
        samples = np.frombuffer(data, dtype=f"<i{bits // 8}") / float(2 ** (bits - 1))
    elif kind == _PCM and bits == 24:
        b = np.frombuffer(data[: len(data) // 3 * 3], dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)) | (
            b[:, 2].astype(np.int8).astype(np.int32) << 16
        )
        samples = ints / float(2**23)
    else:
        raise ValueError(f"{path}: unsupported WAV format {kind} with {bits} bits")
    frames = len(samples) // channels
    return samples[: frames * channels].reshape(frames, channels), int(sample_rate)


__all__ = ["read_wav", "write_wav"]
