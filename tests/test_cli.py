from pathlib import Path

import pytest
import yaml

from acoustic_spaces.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SHOEBOX = EXAMPLES / "shoebox" / "scene.yaml"


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "acoustic_spaces" in out
    assert "compiled core: available" in out


def test_validate_example(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(SHOEBOX)]) == 0
    assert "1 sources, 1 listeners, 12 triangles" in capsys.readouterr().out


def test_validate_reports_problems(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    data["geometry"]["file"] = str(EXAMPLES / "shoebox" / "shoebox.obj")
    del data["assignments"]["walls"]
    data["listeners"][0]["position"] = [6.0, 3.0, 4.0]
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    err = capsys.readouterr().err
    assert "surfaces without a material: walls (84.0 m2)" in err
    assert "listener 'seat-a'" in err


def test_validate_reports_schema_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    data["materials"]["painted_concrete"]["preset"] = "smooth_brickwork"
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    err = capsys.readouterr().err
    assert "materials.painted_concrete: no absorption material named 'smooth_brickwork'" in err


def test_inspect(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(SHOEBOX)]) == 0
    out = capsys.readouterr().out
    assert "volume 144.00 m3" in out
    assert "Sabine" in out and "Eyring" in out
    assert "seat-a" in out and "inside" in out


def test_materials_search(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["materials", "fissured"]) == 0
    assert "ceiling_fissured_tile" in capsys.readouterr().out
    assert main(["materials", "--kind", "scattering"]) == 0
    assert "rpg_qrd" in capsys.readouterr().out
    assert main(["materials", "no-such-material"]) == 1


def test_simulate_writes_responses_and_parameters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from acoustic_spaces.audio import read_wav

    out = tmp_path / "out"
    args = ["simulate", str(SHOEBOX), "--out", str(out), "--rays", "2000", "--length", "0.6"]
    assert main([*args, "--seed", "3", "--sample-rate", "16000", "--normalize"]) == 0
    stdout = capsys.readouterr().out
    assert "talker -> seat-a" in stdout
    assert "T30 (s)" in stdout
    data, fs = read_wav(out / "talker__seat-a.wav")
    assert fs == 16000 and data.shape == (9600, 1)
    assert abs(data).max() == pytest.approx(10 ** (-1 / 20), rel=1e-6)
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["settings"]["seed"] == 3
    (response,) = report["responses"]
    assert response["early_paths"] == 63
    assert len(response["parameters"]["c80"]) == 8
