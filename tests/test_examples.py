import importlib.util
from pathlib import Path

import numpy as np
import pytest
from conftest import skip_without_extra

from acoustic_spaces.cli import main
from acoustic_spaces.engine import simulate
from acoustic_spaces.examples import copy_examples, example_path, list_examples
from acoustic_spaces.room import Room

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NAMES = [e.name for e in list_examples()]


def test_every_example_folder_is_listed() -> None:
    assert NAMES == sorted(p.parent.name for p in EXAMPLES.glob("*/scene.yaml"))
    assert {"church", "concert_hall", "lecture_hall", "shoebox", "studio"} <= set(NAMES)
    for example in list_examples():
        assert example.title and example.description
        assert example.path.is_file()


def test_example_path() -> None:
    assert example_path("studio").parent.name == "studio"
    with pytest.raises(ValueError, match="examples: archway, bim_office, church"):
        example_path("cathedral")


@pytest.mark.parametrize("name", NAMES)
def test_examples_simulate(name: str) -> None:
    skip_without_extra(name)
    room = Room.from_yaml(example_path(name))
    sim = room.scene.simulation
    quick = sim.model_copy(
        update={
            "ir_length_s": 0.3,
            "seed": 1,
            "early": sim.early.model_copy(update={"max_order": 2}),
            "late": sim.late.model_copy(update={"rays": 500}),
        }
    )
    result = simulate(room, settings=quick)
    assert len(result) == len(room.scene.sources) * len(room.scene.listeners)
    for source in room.scene.sources:
        for listener in room.scene.listeners:
            distance = np.linalg.norm(np.subtract(listener.position, source.position))
            delay = result[source.name, listener.name].direct_delay_s
            assert delay == pytest.approx(distance / room.speed_of_sound(), rel=1e-6)


def test_copy_examples(tmp_path: Path) -> None:
    copied = copy_examples(tmp_path, ["concert_hall", "studio"])
    assert copied == [tmp_path / "concert_hall" / "scene.yaml", tmp_path / "studio" / "scene.yaml"]
    # the copy uses its own model file
    room = Room.from_yaml(copied[0])
    assert room.scene.geometry.file == (tmp_path / "concert_hall" / "concert_hall.obj").resolve()
    assert room.problems() == []
    with pytest.raises(FileExistsError):
        copy_examples(tmp_path, ["studio"])
    with pytest.raises(ValueError, match="cathedral"):
        copy_examples(tmp_path / "other", ["cathedral"])


def test_examples_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["examples"]) == 0
    out = capsys.readouterr().out
    for name in NAMES:
        assert f"{name}: " in out
    assert main(["examples", "church"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("church: ") and "studio" not in out
    assert main(["examples", "--copy", str(tmp_path), "studio"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "studio" / "scene.yaml")
    assert main(["examples", "cathedral"]) == 1
    assert "no example named cathedral" in capsys.readouterr().err


def test_listening_room_bass(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("bass", EXAMPLES / "listening_room" / "bass.py")
    bass = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bass)
    room = Room.from_yaml(example_path("listening_room"))
    # The early part at 63 Hz: the wall's reflection in phase against the wall, half a
    # period late a metre out; in energy the wall makes no difference.
    pressure, energy = (bass.levels(room, coherent, 300) for coherent in (125, None))
    wall, out = (pressure[name][1][0] for name in ("against the wall", "a metre out"))
    assert wall - out > 4.0
    wall, out = (energy[name][1][0] for name in ("against the wall", "a metre out"))
    assert abs(wall - out) < 1.0
    assert bass.main(["--rays", "300"]) == 0
    assert "in pressure up to 125 Hz" in capsys.readouterr().out
