"""Head tracking: binaural playback of an Ambisonics response that turns with the
listener's head, so the sources stay where they are in the room.

A simulated AmbiX response holds the sound field at the listener, arriving from every
direction. :class:`HeadTrackedBinaural` runs a dry signal through it in real time,
turns the field against the head's latest orientation once per block (crossfading from
the previous orientation over the block, so turning does not click) and decodes it to
the two ears with :func:`~acoustic_spaces.spatial.decoder.binaural_decoder`.

The orientation lives in a :class:`HeadOrientation`, which anything may set: the
editor's controls, a script, or a head tracker sending OSC over UDP to an
:class:`OscReceiver`. The receiver reads the messages head trackers commonly send,
whatever comes before the last part of the address (``/ypr`` and
``/SceneRotator/ypr`` alike):

- ``.../ypr``: yaw, pitch and roll in degrees;
- ``.../quaternion``, ``.../quaternions`` or ``.../quat``: w, x, y and z;
- ``.../yaw``, ``.../pitch`` or ``.../roll``: one angle in degrees.

Angles follow the scene's convention: yaw turns left, pitch looks up and roll tilts the
head to the right. A tracker that counts an axis the other way round needs that axis in
``invert``.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
from collections.abc import Callable, Iterable

import numpy as np

from acoustic_spaces.render import Convolver
from acoustic_spaces.spatial.decoder import binaural_decoder
from acoustic_spaces.spatial.hrtf import HRTF, spherical_head
from acoustic_spaces.spatial.rotation import (
    FieldRotation,
    angles_matrix,
    matrix_angles,
    quaternion_matrix,
)

#: UDP port :class:`OscReceiver` listens on unless told otherwise.
DEFAULT_OSC_PORT = 9000
#: The axes of a head orientation, as ``invert`` names them.
AXES = ("yaw", "pitch", "roll")
#: Last parts of OSC addresses that carry a quaternion.
QUATERNION_ADDRESSES = ("quaternion", "quaternions", "quat")
#: Deepest nesting of OSC bundles read.
_MAX_BUNDLE_DEPTH = 8


class HeadOrientation:
    """The listener's head orientation, set from one thread and read from another.

    ``invert`` names the axes (``"yaw"``, ``"pitch"``, ``"roll"``) whose incoming
    angles count the other way round. :meth:`centre` takes the current orientation as
    straight ahead; :meth:`rotation` and :meth:`angles` are relative to it.
    """

    def __init__(self, invert: Iterable[str] = ()) -> None:
        self.invert = frozenset(invert)
        unknown = self.invert - set(AXES)
        if unknown:
            raise ValueError(
                f"cannot invert {', '.join(sorted(unknown))}; choose from {', '.join(AXES)}"
            )
        self._lock = threading.Lock()
        self._angles = (0.0, 0.0, 0.0)
        self._raw = np.eye(3)
        self._centre = np.eye(3)
        self._version = 0

    def set_angles(
        self, yaw: float | None = None, pitch: float | None = None, roll: float | None = None
    ) -> None:
        """Yaw, pitch and roll in degrees; an angle left out keeps its last value. Angles
        that are not finite numbers are ignored."""
        with self._lock:
            angles = tuple(
                old if new is None else (-float(new) if axis in self.invert else float(new))
                for axis, old, new in zip(AXES, self._angles, (yaw, pitch, roll), strict=True)
            )
            if not all(math.isfinite(angle) for angle in angles):
                return
            self._angles = angles
            self._raw = angles_matrix(*angles)
            self._version += 1

    def set_quaternion(self, w: float, x: float, y: float, z: float) -> None:
        """The orientation as a quaternion (normalised first) in the scene's axes: x
        forward, y left, z up."""
        if not all(math.isfinite(float(value)) for value in (w, x, y, z)):
            return
        self.set_angles(*matrix_angles(quaternion_matrix(w, x, y, z)))

    def look(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> None:
        """Turn the head to yaw, pitch and roll in degrees relative to the centre, as
        :meth:`angles` gives them back (``invert`` does not apply)."""
        if not all(math.isfinite(float(angle)) for angle in (yaw, pitch, roll)):
            return
        with self._lock:
            self._raw = self._centre @ angles_matrix(yaw, pitch, roll)
            self._angles = matrix_angles(self._raw)
            self._version += 1

    def centre(self) -> None:
        """Take the current orientation as straight ahead."""
        with self._lock:
            self._centre = self._raw.copy()
            self._version += 1

    def rotation(self) -> tuple[int, np.ndarray]:
        """(version, rotation (3, 3)) of the head relative to the centre. The version
        counts the changes, so a reader can tell when to look again."""
        with self._lock:
            return self._version, self._centre.T @ self._raw

    def angles(self) -> tuple[float, float, float]:
        """Yaw, pitch and roll in degrees relative to the centre."""
        return matrix_angles(self.rotation()[1])


def _osc_string(packet: bytes, offset: int) -> tuple[str, int]:
    end = packet.find(b"\0", offset)
    if end < 0:
        raise ValueError("OSC string without its terminating zero")
    return packet[offset:end].decode("utf-8", "replace"), offset + (end - offset) // 4 * 4 + 4


def parse_osc(packet: bytes, _depth: int = 0) -> list[tuple[str, list]]:
    """The messages (address, arguments) of an OSC packet, with bundles unpacked.

    Reads 32-bit floats and integers, doubles, 64-bit integers, strings and the
    argument-less types (true, false, nil, infinity); anything else, or a malformed
    packet, raises ValueError.
    """
    try:
        if packet.startswith(b"#bundle\0"):
            if _depth >= _MAX_BUNDLE_DEPTH:
                raise ValueError("OSC bundles nested too deep")
            messages: list[tuple[str, list]] = []
            offset = 16  # after the tag and the time tag
            while offset < len(packet):
                (size,) = struct.unpack_from(">i", packet, offset)
                offset += 4
                if size < 0 or offset + size > len(packet):
                    raise ValueError("OSC bundle element runs past the packet")
                messages += parse_osc(packet[offset : offset + size], _depth + 1)
                offset += size
            return messages
        address, offset = _osc_string(packet, 0)
        if not address.startswith("/"):
            raise ValueError("not an OSC message")
        if offset >= len(packet):
            return [(address, [])]
        tags, offset = _osc_string(packet, offset)
        if not tags.startswith(","):
            raise ValueError("OSC message without type tags")
        arguments: list = []
        for tag in tags[1:]:
            if tag in "fi":
                arguments.append(struct.unpack_from(f">{tag}", packet, offset)[0])
                offset += 4
            elif tag in "dh":
                arguments.append(
                    struct.unpack_from(">d" if tag == "d" else ">q", packet, offset)[0]
                )
                offset += 8
            elif tag == "s":
                text, offset = _osc_string(packet, offset)
                arguments.append(text)
            elif tag in "TFNI":
                arguments.append({"T": True, "F": False, "N": None, "I": math.inf}[tag])
            else:
                raise ValueError(f"OSC type {tag!r} is not supported")
        return [(address, arguments)]
    except struct.error as exc:
        raise ValueError(f"truncated OSC packet: {exc}") from None


def apply_osc(orientation: HeadOrientation, address: str, arguments: list) -> bool:
    """Set ``orientation`` from one OSC message (see the module's description); False
    when the message is not about the head's orientation."""
    name = address.rstrip("/").rsplit("/", 1)[-1].lower()
    values = [
        float(value)
        for value in arguments
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    if name == "ypr" and len(values) >= 3:
        orientation.set_angles(*values[:3])
    elif name in QUATERNION_ADDRESSES and len(values) >= 4:
        orientation.set_quaternion(*values[:4])
    elif name in AXES and values:
        orientation.set_angles(**{name: values[0]})
    else:
        return False
    return True


class OscReceiver:
    """Sets ``orientation`` from the OSC messages arriving on UDP ``port`` of ``host``
    (0 picks a free port; :attr:`port` says which), in a background thread.

    The default host takes messages from this computer only; ``"0.0.0.0"`` takes them
    from the network too, as from a phone. Use the receiver as a context manager, or
    call :meth:`start` and :meth:`stop`.
    """

    def __init__(
        self, orientation: HeadOrientation, port: int = DEFAULT_OSC_PORT, host: str = "127.0.0.1"
    ) -> None:
        self.orientation = orientation
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind((host, port))
        except OSError:
            self._socket.close()
            raise
        self._socket.settimeout(0.1)
        self.host, self.port = self._socket.getsockname()[:2]
        #: Messages that set the orientation.
        self.received = 0
        #: Packets that were not OSC.
        self.rejected = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="osc-receiver", daemon=True)

    def start(self) -> OscReceiver:
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop listening and free the port."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._socket.close()

    def __enter__(self) -> OscReceiver:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self._socket.recv(65536)
            except (TimeoutError, ConnectionResetError):
                continue
            except OSError:
                break
            try:
                messages = parse_osc(packet)
            except ValueError:
                self.rejected += 1
                continue
            for address, arguments in messages:
                if apply_osc(self.orientation, address, arguments):
                    self.received += 1


class HeadTrackedBinaural:
    """A dry signal through an Ambisonics response, turned against the head and decoded
    to the two ears, one block at a time.

    ``ir`` is (taps, channels) AmbiX (ACN / SN3D) of order 1 to 7, at ``sample_rate``, as
    the simulator renders it for a listener: directions relative to the listener's
    orientation in the scene, which head tracking then turns from. ``hrtf`` defaults
    to the spherical head model and is resampled to ``sample_rate``; ``decoder`` is
    ``"magls"`` or ``"ls"`` (see :func:`~acoustic_spaces.spatial.decoder.binaural_decoder`).
    ``orientation`` (a new :class:`HeadOrientation` by default) is read once per block.
    It works with :func:`acoustic_spaces.render.play` as a :class:`Convolver` does, with
    one input and two outputs, and the only latency is the block.
    """

    inputs = 1
    outputs = 2

    def __init__(
        self,
        ir: np.ndarray,
        sample_rate: float,
        *,
        hrtf: HRTF | None = None,
        orientation: HeadOrientation | None = None,
        block: int = 256,
        decoder: str = "magls",
    ) -> None:
        h = np.asarray(ir, dtype=float)
        if h.ndim != 2 or 0 in h.shape:
            raise ValueError("ir must be (taps, channels) Ambisonics")
        order = math.isqrt(h.shape[1]) - 1
        if order < 1 or (order + 1) ** 2 != h.shape[1]:
            raise ValueError(
                f"head tracking needs Ambisonics of order 1 or more: 4, 9, 16... channels, "
                f"not {h.shape[1]}"
            )
        if hrtf is None:
            hrtf = spherical_head(sample_rate)
        elif hrtf.sample_rate != sample_rate:
            hrtf = hrtf.resampled(sample_rate)
        self.order = order
        self.sample_rate = sample_rate
        self.orientation = orientation if orientation is not None else HeadOrientation()
        self._room = Convolver(h, block)
        self._ears = Convolver(binaural_decoder(order, hrtf, decoder), block)
        self._field = FieldRotation(order)
        self._ramp = (np.arange(1, block + 1, dtype=np.float32) / block)[:, None]
        self._version, rotation = self.orientation.rotation()
        self._matrix = self._turn(rotation)

    @property
    def taps(self) -> int:
        return self._room.taps + self._ears.taps - 1

    @property
    def block(self) -> int:
        return self._room.block

    def _turn(self, rotation: np.ndarray) -> np.ndarray:
        # A source straight ahead of the centred head is at rotation.T @ d once it turns.
        return self._field.matrix(rotation.T).T.astype(np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        """One block (block,) of the dry signal to (block, 2) float32."""
        wet = self._room.process(block)  # (block, channels)
        version, rotation = self.orientation.rotation()
        if version == self._version:
            turned = wet @ self._matrix
        else:
            matrix = self._turn(rotation)
            before = wet @ self._matrix
            turned = before + (wet @ matrix - before) * self._ramp
            self._matrix, self._version = matrix, version
        return self._ears.process(turned)

    def run(
        self,
        signal: np.ndarray,
        head: Callable[[float], tuple[float, float, float]] | None = None,
    ) -> np.ndarray:
        """A whole signal (samples,) through the renderer: (samples + taps - 1, 2).

        ``head`` gives the yaw, pitch and roll in degrees at a time in seconds; it is
        asked at the start of every block. Without it, the orientation stays as it is.
        """
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError("signal must have one channel")
        length = len(x) + self.taps - 1
        blocks = -(-length // self.block)
        padded = np.zeros(blocks * self.block, np.float32)
        padded[: len(x)] = x
        out = []
        for k in range(blocks):
            if head is not None:
                self.orientation.set_angles(*head(k * self.block / self.sample_rate))
            out.append(self.process(padded[k * self.block : (k + 1) * self.block]))
        return np.concatenate(out)[:length]

    def reset(self) -> None:
        """Forget all past input."""
        self._room.reset()
        self._ears.reset()


__all__ = [
    "AXES",
    "DEFAULT_OSC_PORT",
    "QUATERNION_ADDRESSES",
    "HeadOrientation",
    "HeadTrackedBinaural",
    "OscReceiver",
    "apply_osc",
    "parse_osc",
]
