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


@pytest.fixture(autouse=True)
def device_choices(monkeypatch, tmp_path):
    """Keeps the device choices of ``acoustic-spaces tune`` and of auto's trial in the
    test's own folder, not the user's cache."""
    path = tmp_path / "devices.json"
    monkeypatch.setattr("acoustic_spaces.engine.tuning.cache_file", lambda: path)
    return path


@pytest.fixture(autouse=True)
def english(monkeypatch, tmp_path):
    """Runs every test in English, whatever the computer's language, with the language
    picked in the editor kept in the test's own folder."""
    from acoustic_spaces import i18n

    path = tmp_path / "settings.json"
    monkeypatch.setattr(i18n, "settings_file", lambda: path)
    monkeypatch.setenv(i18n.ENV, "en")
    i18n.set_language(None)
    yield path
    i18n.set_language(None)


@pytest.fixture(scope="session")
def two_sources(tmp_path_factory):
    """The shoebox with a second source, 20 dB less powerful than the talker."""
    from pathlib import Path

    import yaml

    from acoustic_spaces.engine import simulate
    from acoustic_spaces.room import Room

    shoebox = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
    scene = yaml.safe_load(shoebox.read_text(encoding="utf-8"))
    scene["geometry"]["file"] = str(shoebox.parent / scene["geometry"]["file"])
    scene["sources"][0]["level_db"] = 94.0
    scene["sources"].append({"name": "radio", "position": [3.0, 2.0, 1.0], "level_db": 74.0})
    path = tmp_path_factory.mktemp("two") / "scene.yaml"
    path.write_text(yaml.safe_dump(scene), encoding="utf-8")
    room = Room.from_yaml(path)
    settings = room.scene.simulation.model_copy(deep=True)
    settings.late.rays = 1_000
    settings.ir_length_s = 0.3
    return simulate(room, settings=settings)


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


def skip_without_extra(name: str) -> None:
    """Skip a test of the example ``name`` when its model needs an extra not installed
    (the wheel tests and the editor's run without the ifc extra)."""
    from acoustic_spaces.examples import list_examples

    missing = next(e.missing_extra for e in list_examples() if e.name == name)
    if missing:
        pytest.skip(f"the {name} example needs the {missing} extra")
