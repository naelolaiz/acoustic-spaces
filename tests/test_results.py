import json
from pathlib import Path

import pytest

from acoustic_spaces.audio import read_wav
from acoustic_spaces.engine import simulate
from acoustic_spaces.results import REPORT_FILE, file_name, response_stem, write_results
from acoustic_spaces.room import Room
from acoustic_spaces.spatial import Ambisonics, Mono

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"


def test_file_names() -> None:
    assert file_name("seat a/1") == "seat_a_1"
    assert file_name("  ") == "unnamed"
    assert response_stem("talker", "seat a") == "talker__seat_a"


def test_write_results(tmp_path: Path) -> None:
    room = Room.from_yaml(SHOEBOX)
    sim = room.scene.simulation
    quick = sim.model_copy(
        update={"ir_length_s": 0.3, "seed": 1, "late": sim.late.model_copy(update={"rays": 500})}
    )
    result = simulate(room, settings=quick)
    messages: list[str] = []
    report = write_results(
        result,
        tmp_path / "out",
        [Mono(), Ambisonics(1)],
        sample_rate=16_000,
        normalize=True,
        progress=messages.append,
    )
    assert report["peak"] == pytest.approx(10 ** (-1 / 20))
    assert json.loads((tmp_path / "out" / REPORT_FILE).read_text()) == report
    entry = report["responses"][0]
    assert [o["wav"] for o in entry["outputs"]] == [
        "talker__seat-a.wav",
        "talker__seat-a.ambix1.wav",
    ]
    data, rate = read_wav(tmp_path / "out" / "talker__seat-a.ambix1.wav")
    assert rate == 16_000 and data.shape[1] == 4
    assert messages and "rendering" in messages[0]
