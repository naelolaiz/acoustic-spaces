from pathlib import Path

import pytest
from pydantic import ValidationError

from acoustic_spaces.materials import (
    OCTAVE_BANDS_HZ,
    builtin_database,
    read_coefficients_csv,
    to_octave_bands,
)
from acoustic_spaces.scene import Material

THIRD_OCTAVES = [
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
    1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
]  # fmt: skip


def test_builtin_database_contents() -> None:
    db = builtin_database()
    assert len(db.entries("absorption")) == 90
    assert len(db.entries("scattering")) == 7
    entry = db.get("rough_concrete")
    assert entry.frequencies[0] == 125
    assert len(entry.frequencies) == len(entry.coefficients)
    assert "pyroomacoustics" in entry.source


def test_unknown_entry_suggests_close_names() -> None:
    with pytest.raises(KeyError, match="rough_concrete"):
        builtin_database().get("rough_concret")


def test_search() -> None:
    hits = builtin_database().search("carpet foam")
    assert {e.id for e in hits} >= {"carpet_6mm_closed_cell_foam", "carpet_6mm_open_cell_foam"}
    assert all(e.kind == "absorption" for e in hits)
    assert builtin_database().search("", kind="scattering")


def test_octave_conversion_averages_third_octaves_and_fills_gaps() -> None:
    values, extrapolated = to_octave_bands([100, 125, 160, 1000, 2000], [0.1, 0.2, 0.3, 0.5, 0.6])
    assert values[OCTAVE_BANDS_HZ.index(125)] == pytest.approx(0.2)
    assert values[0] == pytest.approx(0.2)  # 63 Hz copied from 125 Hz
    assert values[OCTAVE_BANDS_HZ.index(250)] == pytest.approx(0.2)  # nearest octave below
    assert values[-1] == pytest.approx(0.6)
    assert extrapolated == [63, 250, 500, 4000, 8000]


def test_octave_conversion_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        to_octave_bands([], [])
    with pytest.raises(ValueError, match="no data"):
        to_octave_bands([20000], [0.5])


def _bras_csv(path: Path) -> Path:
    alpha = [min(0.01 * k, 1.0) for k in range(len(THIRD_OCTAVES))]
    scatter = [0.3] * len(THIRD_OCTAVES)
    lines = [",".join(str(v) for v in row) for row in (THIRD_OCTAVES, alpha, scatter)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_bras_style_csv(tmp_path: Path) -> None:
    freqs, alpha, scatter = read_coefficients_csv(_bras_csv(tmp_path / "wall.csv"))
    assert freqs == THIRD_OCTAVES
    assert len(alpha) == len(scatter) == 31


def test_csv_with_header_and_semicolons(tmp_path: Path) -> None:
    path = tmp_path / "panel.csv"
    path.write_text("# measured\nfrequency;alpha\n125;250;500\n0.1;0.2;0.3\n", encoding="utf-8")
    freqs, alpha, scatter = read_coefficients_csv(path)
    assert freqs == [125, 250, 500]
    assert alpha == [0.1, 0.2, 0.3]
    assert scatter is None


def test_csv_refuses_values_that_are_not_finite(tmp_path: Path) -> None:
    path = tmp_path / "nan.csv"
    path.write_text("125,250\n0.1,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"nan.csv:2: expected finite numbers"):
        read_coefficients_csv(path)
    path.write_text("125,inf\n0.1,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"nan.csv:1: expected finite numbers"):
        read_coefficients_csv(path)


def test_csv_length_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("125,250\n0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="1 absorption values for 2 frequencies"):
        read_coefficients_csv(path)


# -- materials in a scene -----------------------------------------------------------


def test_preset_material_records_its_source() -> None:
    material = Material(preset="rough_concrete")
    assert material.absorption == pytest.approx([0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.07, 0.07])
    assert material.scattering == [0.1] * len(OCTAVE_BANDS_HZ)
    assert "rough_concrete" in material.source
    assert "63 Hz copied" in material.source


def test_explicit_values_override_a_preset() -> None:
    absorption = [0.5] * len(OCTAVE_BANDS_HZ)
    material = Material(preset="rough_concrete", absorption=absorption, scattering="rpg_qrd")
    assert material.absorption == absorption
    assert material.scattering[OCTAVE_BANDS_HZ.index(500)] == pytest.approx(0.45)
    assert material.source.startswith("scattering: ")


def test_unknown_preset_is_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="did you mean rough_concrete"):
        Material(preset="rough_concret")
    with pytest.raises(ValidationError, match="no scattering material"):
        Material(preset="rough_concrete", scattering="velvet")


def test_material_needs_some_coefficients() -> None:
    with pytest.raises(ValidationError, match="absorption values, a preset or a file"):
        Material()


def test_material_from_file_relative_to_scene(tmp_path: Path) -> None:
    _bras_csv(tmp_path / "wall.csv")
    material = Material.model_validate({"file": "wall.csv"}, context={"base_dir": tmp_path})
    assert material.file == (tmp_path / "wall.csv").resolve()
    # 1 kHz octave: third octaves 800, 1000, 1250 Hz -> indices 16, 17, 18.
    assert material.absorption[OCTAVE_BANDS_HZ.index(1000)] == pytest.approx(0.17)
    assert material.scattering == pytest.approx([0.3] * len(OCTAVE_BANDS_HZ))
    assert material.source == "file wall.csv"


def test_preset_and_file_together_are_rejected(tmp_path: Path) -> None:
    _bras_csv(tmp_path / "wall.csv")
    with pytest.raises(ValidationError, match="not both"):
        Material(preset="rough_concrete", file=tmp_path / "wall.csv")
    with pytest.raises(ValidationError, match="not found"):
        Material(file=tmp_path / "missing.csv")
