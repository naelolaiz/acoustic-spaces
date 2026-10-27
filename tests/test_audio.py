import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.audio import read_wav, write_wav


def test_float_round_trip(tmp_path: Path) -> None:
    data = np.random.default_rng(0).standard_normal((1000, 2)) * 2.0  # above full scale
    write_wav(tmp_path / "x.wav", data, 44_100)
    back, fs = read_wav(tmp_path / "x.wav")
    assert fs == 44_100
    np.testing.assert_allclose(back, data.astype(np.float32), rtol=0, atol=0)


def test_a_long_file_is_written_a_part_at_a_time(tmp_path: Path) -> None:
    # More frames than one part of 4 MiB holds: the same samples back, and the RIFF and
    # data chunks as long as the file has them.
    data = np.random.default_rng(1).standard_normal(((1 << 19) + 7, 3))
    write_wav(tmp_path / "x.wav", data, 48_000)
    back, fs = read_wav(tmp_path / "x.wav")
    assert fs == 48_000
    np.testing.assert_array_equal(back, data.astype(np.float32))
    raw = (tmp_path / "x.wav").read_bytes()
    assert struct.unpack("<I", raw[4:8])[0] == len(raw) - 8
    at = raw.index(b"data")
    assert struct.unpack("<I", raw[at + 4 : at + 8])[0] == data.size * 4 == len(raw) - at - 8


def test_reads_pcm16(tmp_path: Path) -> None:
    samples = (np.sin(np.arange(100) / 5.0) * 20000).astype("<i2")
    with wave.open(str(tmp_path / "pcm.wav"), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(8000)
        fh.writeframes(samples.tobytes())
    back, fs = read_wav(tmp_path / "pcm.wav")
    assert fs == 8000 and back.shape == (100, 1)
    np.testing.assert_allclose(back[:, 0], samples / 32768.0)


def test_rejects_other_files(tmp_path: Path) -> None:
    (tmp_path / "x.wav").write_bytes(b"not a wav file")
    with pytest.raises(ValueError):
        read_wav(tmp_path / "x.wav")
