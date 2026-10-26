import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.audio import read_wav, write_wav
from acoustic_spaces.cli import main
from acoustic_spaces.engine import simulate
from acoustic_spaces.results import (
    REPORT_FILE,
    file_name,
    response_stem,
    unique_stems,
    write_results,
)
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
    # Names that give the same file, also where a file system ignores case.
    assert unique_stems(["A_B", "A_B", "a_b", "unnamed", "A_B-2"]) == [
        "A_B",
        "A_B-2",
        "a_b-3",
        "unnamed",
        "A_B-2-2",
    ]
    # The stem of one file and another's stem with a format's suffix.
    assert unique_stems(["a__b", "a__b.binaural"], [".wav", ".binaural.wav"]) == [
        "a__b",
        "a__b.binaural-2",
    ]


def test_names_that_give_one_file_name_write_files_of_their_own(result, tmp_path: Path) -> None:
    # "A/B" and "A_B" both give A_B, and two HRTF files of one name give one label.
    response = result["talker", "seat-a"]
    twice = dataclasses.replace(
        result, responses={("talker", "A/B"): response, ("talker", "A_B"): response}
    )
    kemar = [Binaural(spherical_head(), label="binaural-kemar") for _ in range(2)]
    report = write_results(twice, tmp_path / "out", [Mono(), *kemar])
    names = [[o["wav"] for o in entry["outputs"]] for entry in report["responses"]]
    assert names == [
        ["talker__A_B.wav", "talker__A_B.binaural-kemar.wav", "talker__A_B.binaural-kemar-2.wav"],
        [
            "talker__A_B-2.wav",
            "talker__A_B-2.binaural-kemar.wav",
            "talker__A_B-2.binaural-kemar-2.wav",
        ],
    ]
    assert sorted(p.name for p in (tmp_path / "out").glob("*.wav")) == sorted(sum(names, []))
    assert [o["format"] for o in report["responses"][0]["outputs"]] == [
        "mono",
        "binaural-kemar",
        "binaural-kemar",
    ]


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


def test_auralize_mixes_the_sources_at_their_power(
    two_sources, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The report gives each source's sound power, as the responses are per unit of it.
    report = write_results(two_sources, tmp_path / "out", [Mono()])
    assert report["sources"] == {
        "talker": {"sound_power_db": 94.0},
        "radio": {"sound_power_db": 74.0},
    }
    click = np.zeros((4800, 1))
    click[0] = 0.5
    write_wav(tmp_path / "dry.wav", click, 48_000)
    talker = read_wav(tmp_path / "out" / "talker__seat-a.wav")[0]
    radio = read_wav(tmp_path / "out" / "radio__seat-a.wav")[0]
    n = max(len(talker), len(radio))
    expected = np.zeros((n + 4799, 1))
    expected[: len(talker)] += 0.5 * talker
    expected[: len(radio)] += 0.05 * radio  # 20 dB less power

    def mix(*extra: str) -> np.ndarray:
        signals = [f"{name}={tmp_path / 'dry.wav'}" for name in ("talker", "radio")]
        out = tmp_path / "mix.wav"
        assert main(["auralize", str(tmp_path / "out"), *signals, "--out", str(out), *extra]) == 0
        return read_wav(out)[0]

    np.testing.assert_allclose(mix(), expected, atol=1e-6)
    expected[: len(radio)] += 0.45 * radio
    np.testing.assert_allclose(mix("--equal-power"), expected, atol=1e-6)
    # Results written before the report gave the powers mix at one power, and say so.
    path = tmp_path / "out" / REPORT_FILE
    old = json.loads(path.read_text(encoding="utf-8"))
    del old["sources"]
    path.write_text(json.dumps(old), encoding="utf-8")
    capsys.readouterr()
    np.testing.assert_allclose(mix(), expected, atol=1e-6)
    assert "simulate again" in capsys.readouterr().err
