import io
import os
import struct
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import audio
from acoustic_spaces.audio import channel_mask, read_wav, write_wav


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


def _rf64(path: Path, data: np.ndarray, monkeypatch) -> bytes:
    """``data`` written to ``path`` as a file over 4 GiB would be."""
    monkeypatch.setattr(audio, "_RIFF_LIMIT", 1000)
    write_wav(path, data, 48_000, 0x60F)
    return path.read_bytes()


def test_a_file_over_4_gib_is_written_as_rf64(tmp_path: Path, monkeypatch) -> None:
    """Past the 4 GiB that RIFF's sizes hold, RF64 (EBU Tech 3306): its ds64 chunk holds
    the RIFF size, the data size and the frames in 64 bits. Read back here, and as BW64
    (ITU-R BS.2088, the same chunks under another name)."""
    data = np.random.default_rng(2).standard_normal((1001, 6))
    write_wav(tmp_path / "small.wav", data, 48_000, 0x60F)
    assert (tmp_path / "small.wav").read_bytes()[:4] == b"RIFF"
    raw = _rf64(tmp_path / "x.wav", data, monkeypatch)
    assert raw[:4] == b"RF64" and raw[4:8] == b"\xff" * 4 and raw[8:16] == b"WAVEds64"
    assert struct.unpack("<QQQI", raw[20:48]) == (len(raw) - 8, data.size * 4, len(data), 0)
    at = raw.index(b"data")
    assert raw[at + 4 : at + 8] == b"\xff" * 4 and len(raw) - at - 8 == data.size * 4
    for name, head in (("x.wav", b"RF64"), ("bw64.wav", b"BW64")):
        (tmp_path / name).write_bytes(head + raw[4:])
        back, fs = read_wav(tmp_path / name)
        assert fs == 48_000 and channel_mask(tmp_path / name) == 0x60F
        np.testing.assert_array_equal(back, data.astype(np.float32))


def test_scipy_reads_the_rf64_files(tmp_path: Path, monkeypatch) -> None:
    """Another reader takes them: scipy, which reads RF64 from 1.14 on."""
    pytest.importorskip("scipy", minversion="1.14")
    from scipy.io import wavfile

    data = np.random.default_rng(3).standard_normal((501, 3))
    _rf64(tmp_path / "x.wav", data, monkeypatch)
    rate, read = wavfile.read(tmp_path / "x.wav")
    assert rate == 48_000
    np.testing.assert_array_equal(read, data.astype(np.float32))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are POSIX's")
def test_reads_a_wav_file_from_a_pipe(tmp_path: Path) -> None:
    data = np.random.default_rng(4).standard_normal((300, 2))
    write_wav(tmp_path / "x.wav", data, 44_100)
    raw = (tmp_path / "x.wav").read_bytes()
    pipe = tmp_path / "pipe.wav"
    for read in (read_wav, channel_mask):
        os.mkfifo(pipe)
        writer = threading.Thread(target=pipe.write_bytes, args=(raw,), daemon=True)
        writer.start()
        found = read(pipe)
        writer.join(timeout=60)
        pipe.unlink()
        if read is read_wav:
            np.testing.assert_array_equal(found[0], data.astype(np.float32))
            assert found[1] == 44_100
        else:
            assert found == channel_mask(tmp_path / "x.wav")


def test_reads_whole_frames_of_a_cut_file(tmp_path: Path) -> None:
    """A data chunk longer than the file, or ending in part of a sample: the whole frames
    it holds."""
    data = np.arange(30, dtype="<i2").reshape(15, 2)
    with wave.open(str(tmp_path / "pcm.wav"), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(8000)
        fh.writeframes(data.tobytes())
    raw = (tmp_path / "pcm.wav").read_bytes()
    (tmp_path / "cut.wav").write_bytes(raw[:-7])
    back, fs = read_wav(tmp_path / "cut.wav")
    assert fs == 8000
    np.testing.assert_array_equal(back, data[:13] / 32768.0)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are POSIX's")
def test_reads_the_channel_mask_of_a_pipe_still_being_written(tmp_path: Path) -> None:
    """channel_mask reads a pipe no further than its fmt chunk: a recorder may still be
    writing the rest."""
    write_wav(tmp_path / "x.wav", np.zeros((10, 6)), 48_000, 0x3F)
    raw = (tmp_path / "x.wav").read_bytes()
    pipe = tmp_path / "pipe.wav"
    os.mkfifo(pipe)
    done = threading.Event()

    def record() -> None:
        with pipe.open("wb") as fh:
            fh.write(raw[:100])
            fh.flush()
            done.wait(30)

    writer = threading.Thread(target=record, daemon=True)
    writer.start()
    started = time.monotonic()
    try:
        assert channel_mask(pipe) == 0x3F
        assert time.monotonic() - started < 20
    finally:
        done.set()
        writer.join(60)


def test_reads_a_file_whose_size_says_0(tmp_path: Path, monkeypatch) -> None:
    """A file of procfs, which says its size is 0 whatever it holds, is read to its end."""

    class Proc(io.BytesIO):
        def seek(self, offset: int, whence: int = 0) -> int:
            return 0 if whence == os.SEEK_END else super().seek(offset, whence)

    data = np.random.default_rng(5).standard_normal((40, 3))
    write_wav(tmp_path / "x.wav", data, 48_000, 0x7)
    raw = (tmp_path / "x.wav").read_bytes()
    monkeypatch.setattr(Path, "open", lambda self, mode="r": Proc(raw))
    back, fs = read_wav(tmp_path / "x.wav")
    mask = channel_mask(tmp_path / "x.wav")
    monkeypatch.undo()
    np.testing.assert_array_equal(back, data.astype(np.float32))
    assert fs == 48_000 and mask == 0x7


def test_refuses_no_channels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="a channel at least"):
        write_wav(tmp_path / "x.wav", np.zeros((5, 0)), 48_000)
    write_wav(tmp_path / "x.wav", np.zeros((5, 1)), 48_000)
    raw = bytearray((tmp_path / "x.wav").read_bytes())
    raw[22:24] = struct.pack("<H", 0)  # the fmt chunk's channels
    (tmp_path / "none.wav").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="no channels"):
        read_wav(tmp_path / "none.wav")
