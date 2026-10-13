import sys
import types

import numpy as np
import pytest


class _FakeStream:
    """Runs the callback block by block on entry, as a sound card would."""

    def __init__(self, module, channels, callback, blocksize, finished_callback, **_):
        self.module, self.callback, self.block = module, callback, blocksize
        self.channels, self.finished = channels, finished_callback

    def __enter__(self):
        duplex = isinstance(self.channels, tuple)
        outputs = self.channels[1] if duplex else self.channels
        status = [True] + [False] * 10_000  # one dropout at the start
        try:
            for k in range(10_000):
                out = np.full((self.block, outputs), np.nan, np.float32)
                if duplex:
                    data = self.module.live_input[k * self.block : (k + 1) * self.block]
                    data = np.pad(data, ((0, self.block - len(data)), (0, 0)))
                    self.callback(data, out, self.block, None, status[k])
                else:
                    self.callback(out, self.block, None, status[k])
                self.module.played.append(out)
        except self.module.CallbackStop:
            self.module.played.append(out)
        self.finished()
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def fake_sounddevice(monkeypatch):
    """A stand-in for the sounddevice module: streams play into ``played``."""
    module = types.ModuleType("sounddevice")
    module.CallbackStop = type("CallbackStop", (Exception,), {})
    module.played = []
    module.live_input = np.zeros((0, 1), np.float32)
    module.OutputStream = lambda **kw: _FakeStream(module, **kw)
    module.Stream = lambda **kw: _FakeStream(module, **kw)
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module
