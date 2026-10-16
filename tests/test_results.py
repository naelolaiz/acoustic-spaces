import json
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.audio import read_wav, write_wav
from acoustic_spaces.cli import main
from acoustic_spaces.engine import simulate
from acoustic_spaces.results import REPORT_FILE, file_name, response_stem, write_results
from acoustic_spaces.room import Room
from acoustic_spaces.spatial import Ambisonics, Binaural, Mono, spherical_head

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"


@pytest.fixture(scope="module")
def result():
    room = Room.from_yaml(SHOEBOX)
    sim = room.scene.simulation
    quick = sim.model_copy(
        update={"ir_length_s": 0.3, "seed": 1, "late": sim.late.model_copy(update={"rays": 500})}
    )
    return simulate(room, settings=quick)


def test_file_names() -> None:
    assert file_name("seat a/1") == "seat_a_1"
    assert file_name("  ") == "unnamed"
    assert response_stem("talker", "seat a") == "talker__seat_a"


def test_write_results(result, tmp_path: Path) -> None:
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


def test_a_binaural_output_through_a_sofa_file(
    result, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A scene's binaural output with an hrtf is labelled binaural-<file name>: it gives
    # IACC as the spherical head's does, and auralize --format binaural finds it.
    ears = Binaural(spherical_head(), label="binaural-kemar")
    report = write_results(result, tmp_path / "out", [Mono(), ears])
    entry = report["responses"][0]
    assert entry["outputs"][1]["wav"] == "talker__seat-a.binaural-kemar.wav"
    assert all(0.0 < v <= 1.0 for v in entry["parameters"]["iacc_e"])
    click = np.zeros((4800, 1))
    click[0] = 0.5
    write_wav(tmp_path / "dry.wav", click, 48_000)
    mix = tmp_path / "mix.wav"
    args = ["auralize", str(tmp_path / "out"), f"talker={tmp_path / 'dry.wav'}"]
    assert main([*args, "--format", "binaural", "--out", str(mix)]) == 0
    assert "talker__seat-a.binaural-kemar.wav" in capsys.readouterr().err
    assert read_wav(mix)[0].shape[1] == 2
