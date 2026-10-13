from pathlib import Path

import pytest

from acoustic_spaces.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "acoustic_spaces" in out
    assert "compiled core: available" in out


def test_validate_example(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(EXAMPLES / "shoebox" / "scene.yaml")]) == 0
    assert "1 sources, 1 listeners" in capsys.readouterr().out
