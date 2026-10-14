import math
import socket
import struct
import time

import numpy as np
import pytest

from acoustic_spaces.headtracking import (
    HeadOrientation,
    HeadTrackedBinaural,
    OscReceiver,
    apply_osc,
    parse_osc,
)
from acoustic_spaces.render import convolve, play
from acoustic_spaces.spatial import Ambisonics, from_angles, real_spherical_harmonics
from acoustic_spaces.spatial.decoder import binaural_decoder
from acoustic_spaces.spatial.hrtf import spherical_head
from acoustic_spaces.spatial.rotation import (
    FieldRotation,
    angles_matrix,
    matrix_angles,
    quaternion_matrix,
)

FS = 48_000


@pytest.fixture(scope="module")
def head():
    return spherical_head(FS, directions=512)


def _about(axis, degrees: float) -> np.ndarray:
    """The rotation about a unit axis, by the right-hand rule."""
    half = math.radians(degrees) / 2
    return quaternion_matrix(math.cos(half), *(math.sin(half) * np.asarray(axis, float)))


def _osc(address: str, *values, tags: str | None = None) -> bytes:
    def text(value: str) -> bytes:
        raw = value.encode() + b"\0"
        return raw + b"\0" * (-len(raw) % 4)

    tags = tags or "f" * len(values)
    formats = {"i": ">i", "f": ">f", "d": ">d", "h": ">q"}
    packed = b""
    for tag, value in zip(tags, values, strict=False):
        packed += text(value) if tag == "s" else struct.pack(formats[tag], value)
    return text(address) + text("," + tags) + packed


def _bundle(*messages: bytes) -> bytes:
    body = b"".join(struct.pack(">i", len(m)) + m for m in messages)
    return b"#bundle\0" + b"\0" * 7 + b"\1" + body


def test_rotation_matrices_and_angles() -> None:
    rng = np.random.default_rng(3)
    for yaw, pitch, roll in rng.uniform([-180, -89, -180], [180, 89, 180], (20, 3)):
        r = angles_matrix(yaw, pitch, roll)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(matrix_angles(r), (yaw, pitch, roll), atol=1e-9)
    # Yaw turns left (about +z), pitch looks up (about -y), roll tilts right (about +x).
    np.testing.assert_allclose(angles_matrix(yaw=30), _about([0, 0, 1], 30), atol=1e-12)
    np.testing.assert_allclose(angles_matrix(pitch=20), _about([0, -1, 0], 20), atol=1e-12)
    np.testing.assert_allclose(angles_matrix(roll=40), _about([1, 0, 0], 40), atol=1e-12)
    np.testing.assert_allclose(angles_matrix(pitch=90) @ [1, 0, 0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(quaternion_matrix(0, 0, 0, 0), np.eye(3))
    np.testing.assert_allclose(quaternion_matrix(2, 0, 0, 0), np.eye(3))


def test_field_rotation_turns_plane_waves() -> None:
    rng = np.random.default_rng(4)
    directions = rng.standard_normal((30, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for order in (0, 1, 3, 7):
        field = FieldRotation(order)
        a = angles_matrix(*rng.uniform(-180, 180, 3))
        b = angles_matrix(*rng.uniform(-180, 180, 3))
        m = field.matrix(a)
        encoded = real_spherical_harmonics(order, directions)  # (dirs, channels)
        turned = real_spherical_harmonics(order, directions @ a.T)
        np.testing.assert_allclose(encoded @ m.T, turned, atol=1e-9)
        np.testing.assert_allclose(m @ m.T, np.eye(field.channels), atol=1e-9)
        np.testing.assert_allclose(field.matrix(a @ b), m @ field.matrix(b), atol=1e-9)
        np.testing.assert_allclose(field.matrix(np.eye(3)), np.eye(field.channels), atol=1e-9)
    with pytest.raises(ValueError, match="order must be 0 to 7"):
        FieldRotation(8)


def _bands(decoder: np.ndarray, hrtf, order: int):
    """Level error per octave (dB) over all directions and the rms ILD error above
    1.5 kHz (dB) of plane waves decoded through ``decoder``."""
    directions = hrtf.directions[::3]
    n = 8192
    decoded = np.einsum(
        "dc,bce->deb",
        real_spherical_harmonics(order, directions),
        np.fft.rfft(decoder, n, axis=0),
    )
    target = np.fft.rfft(hrtf.hrir[::3], n, axis=-1)
    f = np.fft.rfftfreq(n, 1 / hrtf.sample_rate)
    levels = []
    for band in (250, 500, 1000, 2000, 4000, 8000, 16000):
        m = (f >= band / math.sqrt(2)) & (f < band * math.sqrt(2))
        levels.append(
            10 * math.log10((abs(decoded[..., m]) ** 2).mean() / (abs(target[..., m]) ** 2).mean())
        )
    m = (f >= 1500) & (f < 16000)

    def ild(x):
        return 10 * np.log10((abs(x[:, 0, m]) ** 2).mean(-1) / (abs(x[:, 1, m]) ** 2).mean(-1))

    return np.array(levels), float(np.sqrt(((ild(decoded) - ild(target)) ** 2).mean()))


def test_binaural_decoder_keeps_levels_and_level_differences(head) -> None:
    for order in (1, 3):
        magls = binaural_decoder(order, head)
        assert magls.shape == (512, (order + 1) ** 2, 2)
        levels, ild_error = _bands(magls, head, order)
        assert np.all(abs(levels) < 0.5), levels
        assert ild_error < (1.5 if order == 1 else 1.0)
        # Least squares loses the highs and the level differences between the ears.
        levels, ls_error = _bands(binaural_decoder(order, head, "ls"), head, order)
        assert levels[-1] < -10 and ls_error > 3 * ild_error
        # The filters hold all their energy, causal and early on.
        energy = (magls**2).sum(axis=(1, 2))
        assert energy[: len(energy) // 4].sum() > 0.99 * energy.sum()
    with pytest.raises(ValueError, match="order must be 0 to 7"):
        binaural_decoder(9, head)
    with pytest.raises(ValueError, match="unknown decoder 'x'"):
        binaural_decoder(1, head, "x")


def test_head_orientation() -> None:
    head = HeadOrientation()
    version, rotation = head.rotation()
    np.testing.assert_allclose(rotation, np.eye(3))
    head.set_angles(30, 10)
    head.set_angles(roll=5)  # the others stay
    np.testing.assert_allclose(head.angles(), (30, 10, 5), atol=1e-9)
    assert head.rotation()[0] == version + 2
    head.set_angles(yaw=math.nan)
    assert head.rotation()[0] == version + 2
    head.set_quaternion(math.cos(math.radians(20)), 0, 0, math.sin(math.radians(20)))
    np.testing.assert_allclose(head.angles(), (40, 0, 0), atol=1e-9)
    head.centre()
    np.testing.assert_allclose(head.angles(), (0, 0, 0), atol=1e-9)
    head.set_angles(yaw=50)  # 10 degrees left of the centre
    np.testing.assert_allclose(head.angles(), (10, 0, 0), atol=1e-9)

    head.look(-20, 15)  # relative to the centre
    np.testing.assert_allclose(head.angles(), (-20, 15, 0), atol=1e-9)
    head.set_angles(roll=4)  # a tracker's partial update keeps the rest
    np.testing.assert_allclose(head.angles()[1:], (15, 4), atol=1e-9)
    version = head.rotation()[0]
    head.look(math.inf)
    assert head.rotation()[0] == version

    inverted = HeadOrientation(invert=["yaw", "roll"])
    inverted.set_angles(30, 10, 5)
    np.testing.assert_allclose(inverted.angles(), (-30, 10, -5), atol=1e-9)
    with pytest.raises(ValueError, match="cannot invert nod; choose from yaw, pitch, roll"):
        HeadOrientation(invert=["nod"])


def test_osc_messages() -> None:
    assert parse_osc(_osc("/ypr", 10.0, -5.0, 2.5)) == [("/ypr", [10.0, -5.0, 2.5])]
    message = _osc("/x", 3, 1.5, 7, "hi", tags="idhs")
    assert parse_osc(message) == [("/x", [3, 1.5, 7, "hi"])]
    assert parse_osc(_osc("/x", tags="TFN")) == [("/x", [True, False, None])]
    assert parse_osc(b"/ping\0\0\0") == [("/ping", [])]
    nested = _bundle(_osc("/yaw", 1.0), _bundle(_osc("/pitch", 2.0)))
    assert parse_osc(nested) == [("/yaw", [1.0]), ("/pitch", [2.0])]
    deep = _osc("/yaw", 1.0)
    for _ in range(9):
        deep = _bundle(deep)
    for bad, match in (
        (b"garbage", "terminating zero"),
        (b"ypr\0", "not an OSC message"),
        (b"/ypr\0\0\0\0ff\0\0", "without type tags"),
        (_osc("/ypr", 1.0, 2.0)[:-2], "truncated"),
        (_osc("/x", tags="b"), "type 'b' is not supported"),
        (b"#bundle\0" + b"\0" * 8 + struct.pack(">i", 99), "runs past"),
        (deep, "nested too deep"),
    ):
        with pytest.raises(ValueError, match=match):
            parse_osc(bad)

    head = HeadOrientation()
    assert apply_osc(head, "/SceneRotator/ypr", [20.0, 5.0, 0.0])
    assert apply_osc(head, "/head/roll/", [3])
    np.testing.assert_allclose(head.angles(), (20, 5, 3), atol=1e-9)
    half = math.radians(15)
    assert apply_osc(head, "/SceneRotator/quaternions", [math.cos(half), 0, 0, math.sin(half)])
    np.testing.assert_allclose(head.angles(), (30, 0, 0), atol=1e-9)
    assert not apply_osc(head, "/ypr", [1.0, 2.0])  # too few angles
    assert not apply_osc(head, "/yaw", [True])
    assert not apply_osc(head, "/volume", [0.5])


def test_osc_receiver_over_udp() -> None:
    head = HeadOrientation()
    with (
        OscReceiver(head, port=0) as receiver,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as out,
    ):
        assert receiver.host == "127.0.0.1" and receiver.port > 0
        out.sendto(b"not osc", (receiver.host, receiver.port))
        out.sendto(
            _bundle(_osc("/ypr", 45.0, 0.0, 0.0), _osc("/pitch", -10.0)),
            (receiver.host, receiver.port),
        )
        deadline = time.monotonic() + 10
        while receiver.received < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert receiver.received == 2 and receiver.rejected == 1
    np.testing.assert_allclose(head.angles(), (45, -10, 0), atol=1e-6)
    with OscReceiver(head, port=0) as receiver, pytest.raises(OSError):
        OscReceiver(head, port=receiver.port)


def _plane_wave(order: int, yaw: float, taps: int = 200) -> np.ndarray:
    ir = np.zeros((taps, (order + 1) ** 2))
    ir[10] = Ambisonics(order).gains(from_angles(yaw, 0.0)[None])[0]
    return ir


def _ild(x: np.ndarray) -> float:
    return float(10 * np.log10((x[:, 0] ** 2).sum() / (x[:, 1] ** 2).sum()))


def test_head_tracked_binaural(head) -> None:
    ir = _plane_wave(3, 90.0)  # a source on the left
    renderer = HeadTrackedBinaural(ir, FS, hrtf=head, block=128)
    assert (renderer.inputs, renderer.outputs, renderer.block, renderer.order) == (1, 2, 128, 3)
    assert renderer.taps == 200 + 512 - 1
    noise = np.random.default_rng(1).standard_normal(4_800)
    out = renderer.run(noise)
    # Without turning, the same as convolving with the response, then the decoder.
    wet = convolve(noise, ir)
    decoder = binaural_decoder(3, head)
    expected = sum(convolve(wet[:, c], decoder[:, c]) for c in range(16))
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, atol=1e-4 * abs(expected).max())
    assert _ild(out) > 10
    for yaw, ild in ((90, 0.0), (180, -_ild(out)), (-90, 0.0)):
        renderer.reset()
        renderer.orientation.set_angles(yaw=yaw)
        assert abs(_ild(renderer.run(noise)) - ild) < 0.5, yaw
    # A head that starts turned plays turned from the first block.
    turned = HeadOrientation()
    turned.set_angles(yaw=180)
    first = HeadTrackedBinaural(ir, FS, hrtf=head, block=128, orientation=turned).run(noise)
    assert _ild(first[:256]) < -10

    with pytest.raises(ValueError, match="order 1 or more: 4, 9, 16... channels, not 5"):
        HeadTrackedBinaural(np.zeros((10, 5)), FS)
    with pytest.raises(ValueError, match="one channel"):
        renderer.run(np.zeros((10, 2)))
    resampled = HeadTrackedBinaural(_plane_wave(1, 0.0), 44_100, hrtf=head)
    assert resampled.run(noise[:100]).shape == (100 + resampled.taps - 1, 2)


def test_turning_crossfades(head) -> None:
    """Swapping the ears at once does not step: the block crossfades."""
    tone = np.sin(2 * math.pi * 2_000 * np.arange(9_600) / FS)
    renderer = HeadTrackedBinaural(_plane_wave(1, 90.0), FS, hrtf=head, block=256)
    steady = renderer.run(tone)
    renderer.reset()
    jumped = renderer.run(tone, head=lambda t: (180.0 if t >= 0.1 else 0.0, 0.0, 0.0))
    np.testing.assert_allclose(jumped[:4_800], steady[:4_800], atol=1e-5)
    slope = abs(np.diff(steady[1_000:4_000], axis=0)).max()
    assert abs(np.diff(jumped[4_000:6_000], axis=0)).max() < 1.5 * slope
    assert _ild(steady[6_000:9_000]) > 5
    assert _ild(jumped[6_000:9_000]) < -5


def test_play_follows_the_head(fake_sounddevice, head) -> None:
    ir = _plane_wave(1, 90.0)
    signal = np.random.default_rng(2).standard_normal(1_000)
    turned = HeadOrientation()
    turned.set_angles(yaw=45)
    renderer = HeadTrackedBinaural(ir, FS, hrtf=head, block=128, orientation=turned)
    report = play(renderer, FS, signal, block=999, gain=0.5)
    played = np.concatenate(fake_sounddevice.played)
    renderer.reset()
    expected = 0.5 * renderer.run(signal)
    assert report.frames == len(played) == 128 * -(-len(expected) // 128)
    np.testing.assert_allclose(played[: len(expected)], expected, atol=1e-5)
