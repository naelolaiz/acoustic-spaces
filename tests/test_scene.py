from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from acoustic_spaces.scene import OCTAVE_BANDS_HZ, Material, Scene

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_example_scene_loads() -> None:
    scene = Scene.from_yaml(EXAMPLES / "shoebox" / "scene.yaml")

    assert scene.name == "Shoebox 8 x 6 x 3 m"
    assert scene.geometry.file.is_absolute()
    assert scene.geometry.file.exists()
    assert set(scene.assignments.values()) <= set(scene.materials)
    assert [s.name for s in scene.sources] == ["talker"]
    assert [listener.name for listener in scene.listeners] == ["seat-a"]


def test_scalar_scattering_is_expanded_per_band() -> None:
    material = Material(absorption=[0.1] * len(OCTAVE_BANDS_HZ), scattering=0.2)
    assert material.scattering == [0.2] * len(OCTAVE_BANDS_HZ)


def test_wrong_band_count_is_rejected() -> None:
    with pytest.raises(ValidationError, match="absorption"):
        Material(absorption=[0.1, 0.2, 0.3])


def test_unknown_material_assignment_is_rejected() -> None:
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    data["assignments"]["floor"] = "velvet"
    with pytest.raises(ValidationError, match="velvet"):
        Scene.model_validate(data)


def test_round_trip(tmp_path: Path) -> None:
    scene = Scene.from_yaml(EXAMPLES / "shoebox" / "scene.yaml")
    out = tmp_path / "copy.yaml"
    scene.to_yaml(out)
    again = Scene.from_yaml(out)
    assert again.model_dump() == scene.model_dump()


def test_orientation_forward() -> None:
    from acoustic_spaces.scene.model import Orientation

    assert Orientation().forward() == pytest.approx((1.0, 0.0, 0.0))
    assert Orientation(yaw=90).forward() == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert Orientation(yaw=180, pitch=90).forward() == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


def test_default_material_must_exist() -> None:
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    data["default_material"] = "velvet"
    with pytest.raises(ValidationError, match="velvet"):
        Scene.model_validate(data)


def test_example_materials_come_from_presets() -> None:
    scene = Scene.from_yaml(EXAMPLES / "shoebox" / "scene.yaml")
    for material in scene.materials.values():
        assert material.preset is not None
        assert len(material.absorption) == len(OCTAVE_BANDS_HZ)
        assert "pyroomacoustics" in material.source
