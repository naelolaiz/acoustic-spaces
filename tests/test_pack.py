"""Packs: a scene and every file it names in one zip, to run on another computer."""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from acoustic_spaces import __version__
from acoustic_spaces.audio import write_wav
from acoustic_spaces.cli import main
from acoustic_spaces.pack import MANIFEST_FILE, SCENE_FILE, pack, unpack
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Scene

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _omni_csv(path: Path) -> None:
    """A directivity table of 0 dB in every direction."""
    lines = ["azimuth, elevation, 125 Hz, 1000 Hz"]
    lines += [f"{az}, {el}, 0, 0" for el in range(-90, 91, 30) for az in range(0, 360, 30)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scattered_scene(root: Path) -> Path:
    """A scene whose files lie under its folder (a glTF model and its buffers) and
    outside it (a material library, a directivity array and its element's table, the
    WAV file the source plays)."""
    project, shared, arrays = root / "project", root / "shared", root / "arrays"
    (project / "models").mkdir(parents=True)
    for folder in (shared, arrays):
        folder.mkdir()
    box = trimesh.Scene(trimesh.creation.box(extents=(4.0, 3.0, 2.5)))
    for name, data in box.export(file_type="gltf").items():
        (project / "models" / name).write_bytes(data)
    shutil.copy(EXAMPLES / "material_libraries" / "catt_materials.geo", shared)
    _omni_csv(arrays / "element.csv")
    write_wav(shared / "talk.wav", np.zeros(4800), 48000)
    (arrays / "column.yaml").write_text(
        "element: {file: element.csv}\ncount: 4\nspacing_m: 0.2\n", encoding="utf-8"
    )
    scene = {
        "name": "Scattered files",
        "geometry": {"file": "models/model.gltf"},
        "materials": {"plaster": {"file": "../shared/catt_materials.geo", "entry": "walls"}},
        "default_material": "plaster",
        "sources": [
            {
                "name": "column",
                "position": [0.3, 0.2, 0.1],
                "directivity": {"file": str(arrays / "column.yaml")},
                "signal": "../shared/talk.wav",
            }
        ],
        "listeners": [{"name": "seat", "position": [-0.8, -0.4, 0.0]}],
        "simulation": {"late": {"rays": 200}, "ir_length_s": 0.3},
    }
    path = project / "scene.yaml"
    path.write_text(yaml.safe_dump(scene), encoding="utf-8")
    return path


def test_pack_runs_as_the_scene_does(tmp_path: Path, capsys) -> None:
    scene = EXAMPLES / "hillside" / "scene.yaml"
    zipped = tmp_path / "hillside.zip"
    assert main(["pack", str(scene), str(zipped)]) == 0
    out = capsys.readouterr().out
    assert f"wrote {zipped}: the scene and the 2 files it names" in out
    assert "hill.asc" in out and "houses.geojson" in out
    with zipfile.ZipFile(zipped) as archive:
        assert sorted(archive.namelist()) == sorted(
            [SCENE_FILE, MANIFEST_FILE, "hill.asc", "houses.geojson"]
        )
        manifest = json.loads(archive.read(MANIFEST_FILE))
    assert manifest["acoustic_spaces"] == __version__
    assert manifest["scene"] == "Yard behind a hill"

    runs = {}
    for name, source in (("scene", scene), ("pack", zipped)):
        out_dir = tmp_path / name
        options = ["--rays", "300", "--seed", "3", "--threads", "1"]
        assert main(["simulate", str(source), "--out", str(out_dir), *options]) == 0
        runs[name] = out_dir
    capsys.readouterr()
    wavs = sorted(p.name for p in runs["scene"].glob("*.wav"))
    assert wavs and wavs == sorted(p.name for p in runs["pack"].glob("*.wav"))
    for wav in wavs:
        assert (runs["scene"] / wav).read_bytes() == (runs["pack"] / wav).read_bytes()


def test_files_from_elsewhere_come_along(tmp_path: Path, capsys) -> None:
    path = _scattered_scene(tmp_path / "source")
    zipped = tmp_path / "scattered.zip"
    assert main(["pack", str(path), str(zipped)]) == 0
    capsys.readouterr()
    with zipfile.ZipFile(zipped) as archive:
        names = set(archive.namelist())
        written = yaml.safe_load(archive.read(SCENE_FILE))
    assert names == {
        SCENE_FILE,
        MANIFEST_FILE,
        "models/model.gltf",
        "models/gltf_buffer_0.bin",
        "models/gltf_buffer_1.bin",
        "external/shared/catt_materials.geo",
        "external/shared/talk.wav",
        "external/arrays/column.yaml",
        "external/arrays/element.csv",
    }
    assert written["geometry"]["file"] == "models/model.gltf"
    assert written["materials"]["plaster"]["file"] == "external/shared/catt_materials.geo"
    assert written["sources"][0]["directivity"]["file"] == "external/arrays/column.yaml"
    assert written["sources"][0]["signal"] == "external/shared/talk.wav"

    # The pack alone is enough: the originals can go.
    original = Room.from_yaml(path)
    shutil.rmtree(tmp_path / "source")
    unpacked = unpack(zipped, tmp_path / "unpacked")
    assert unpacked.version == __version__
    room = Room.from_yaml(unpacked.scene)
    np.testing.assert_array_equal(room.mesh.vertices, original.mesh.vertices)
    plaster = room.scene.materials["plaster"]
    assert plaster.absorption == original.scene.materials["plaster"].absorption
    assert main(["validate", str(zipped)]) == 0


def test_a_scene_never_saved_packs_every_file_apart(tmp_path: Path) -> None:
    scene = Scene.from_yaml(EXAMPLES / "material_libraries" / "scene.yaml")
    packed = pack(scene, tmp_path / "seminar.zip")
    assert [name for name, _bytes in packed.files] == [
        "external/material_libraries/catt_materials.geo",
        "external/material_libraries/odeon_materials.li8",
    ]
    assert packed.size == sum(
        (EXAMPLES / "material_libraries" / name).stat().st_size
        for name in ("catt_materials.geo", "odeon_materials.li8")
    )


def test_folders_of_the_same_name_stay_apart(tmp_path: Path) -> None:
    data = yaml.safe_load((EXAMPLES / "material_libraries" / "scene.yaml").read_text())
    for kind in ("a", "b"):
        folder = tmp_path / kind / "libraries"
        folder.mkdir(parents=True)
        shutil.copy(EXAMPLES / "material_libraries" / "catt_materials.geo", folder)
    data["materials"]["carpet"]["file"] = str(tmp_path / "a" / "libraries" / "catt_materials.geo")
    data["materials"]["plasterboard"]["file"] = str(
        tmp_path / "b" / "libraries" / "catt_materials.geo"
    )
    data["materials"]["ceiling_tiles"] = {"preset": "linoleum_on_concrete"}
    path = tmp_path / "scene" / "scene.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    packed = pack(Scene.from_yaml(path), tmp_path / "two.zip", base_dir=path.parent)
    assert [name for name, _bytes in packed.files] == [
        "external/libraries/catt_materials.geo",
        "external/libraries-2/catt_materials.geo",
    ]


def test_a_pack_that_fails_keeps_the_earlier_one(tmp_path: Path, monkeypatch) -> None:
    scene = Scene.from_yaml(EXAMPLES / "material_libraries" / "scene.yaml")
    out = tmp_path / "seminar.zip"
    out.write_bytes(b"an earlier pack")

    def disk_full(path: Path) -> str:
        raise OSError(28, "No space left on device")

    # Failing while it writes, after every file the scene names was found.
    monkeypatch.setattr("acoustic_spaces.pack._sha256", disk_full)
    with pytest.raises(OSError, match="No space left"):
        pack(scene, out)
    assert out.read_bytes() == b"an earlier pack"
    assert list(tmp_path.iterdir()) == [out]
    monkeypatch.undo()
    pack(scene, out)
    assert zipfile.is_zipfile(out)
    assert list(tmp_path.iterdir()) == [out]


def test_a_file_naming_one_outside_its_folder_cannot_be_packed(tmp_path: Path) -> None:
    path = _scattered_scene(tmp_path)
    column = tmp_path / "arrays" / "column.yaml"
    column.write_text(
        "element: {file: ../shared/element.csv}\ncount: 2\nspacing_m: 0.2\n", encoding="utf-8"
    )
    _omni_csv(tmp_path / "shared" / "element.csv")
    with pytest.raises(ValueError, match="column.yaml names .*element.csv, outside its folder"):
        pack(Scene.from_yaml(path), tmp_path / "out.zip", base_dir=path.parent)
    assert not (tmp_path / "out.zip").exists()


def test_pack_refuses_what_cannot_run(tmp_path: Path, capsys) -> None:
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"]["file"] = str(EXAMPLES / "shoebox" / "shoebox.obj")
    data["listeners"][0]["position"] = [6.0, 3.0, 4.0]
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert main(["pack", str(path), str(tmp_path / "out.zip")]) == 1
    assert "listener 'seat-a'" in capsys.readouterr().err
    assert not (tmp_path / "out.zip").exists()

    shoebox = str(EXAMPLES / "shoebox" / "scene.yaml")
    assert main(["pack", shoebox, str(tmp_path / "out.tar")]) == 1
    assert "a pack is a zip file: name it out.zip" in capsys.readouterr().err
    assert main(["pack", shoebox, str(tmp_path / "out.zip")]) == 0
    assert main(["pack", str(tmp_path / "out.zip"), str(tmp_path / "again.zip")]) == 1
    assert "is a pack already" in capsys.readouterr().err


def _rewrite(zipped: Path, change) -> None:
    """Rewrite a pack, ``change(name, data)`` giving each file's new data (None: left out)."""
    with zipfile.ZipFile(zipped) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(zipped, "w") as archive:
        for name, data in files.items():
            data = change(name, data)
            if data is not None:
                archive.writestr(name, data)


def test_unpack_checks_what_it_unpacks(tmp_path: Path, capsys) -> None:
    zipped = tmp_path / "hillside.zip"
    pack(Scene.from_yaml(EXAMPLES / "hillside" / "scene.yaml"), zipped, EXAMPLES / "hillside")

    damaged = tmp_path / "damaged.zip"
    shutil.copy(zipped, damaged)
    _rewrite(damaged, lambda name, data: data + b"0" if name == "hill.asc" else data)
    with pytest.raises(ValueError, match="hill.asc is damaged"):
        unpack(damaged, tmp_path / "d")
    assert main(["simulate", str(damaged), "--out", str(tmp_path / "r")]) == 1
    assert "error: damaged.zip: hill.asc is damaged" in capsys.readouterr().err

    def manifest(change):
        def rewrite(name, data):
            if name != MANIFEST_FILE:
                return data
            content = json.loads(data)
            change(content)
            return json.dumps(content)

        return rewrite

    for change, message in (
        (lambda m: m["files"].update({"../escape.txt": m["files"]["hill.asc"]}), "not a name"),
        (lambda m: m["files"].update({"/etc/x": m["files"]["hill.asc"]}), "not a name"),
        (lambda m: m["files"].update({"missing.asc": "0"}), "lacks missing.asc"),
        (lambda m: m.update({"format": 2, "acoustic_spaces": "99.0"}), "update to that version"),
        (lambda m: m["files"].pop(SCENE_FILE), "lists no scene.yaml"),
    ):
        bad = tmp_path / "bad.zip"
        shutil.copy(zipped, bad)
        _rewrite(bad, manifest(change))
        with pytest.raises(ValueError, match=message):
            unpack(bad, tmp_path / "b")
    assert not (tmp_path / "escape.txt").exists()

    no_manifest = tmp_path / "plain.zip"
    shutil.copy(zipped, no_manifest)
    _rewrite(no_manifest, lambda name, data: None if name == MANIFEST_FILE else data)
    with pytest.raises(ValueError, match="plain.zip is not a pack: it has no pack.json"):
        unpack(no_manifest, tmp_path / "p")
    text = tmp_path / "text.zip"
    text.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ValueError, match="text.zip is not a pack"):
        unpack(text, tmp_path / "t")


def test_a_pack_from_another_version_warns(tmp_path: Path, capsys) -> None:
    zipped = tmp_path / "shoebox.zip"
    assert main(["pack", str(EXAMPLES / "shoebox" / "scene.yaml"), str(zipped)]) == 0

    def older(name, data):
        if name != MANIFEST_FILE:
            return data
        return json.dumps({**json.loads(data), "acoustic_spaces": "0.9.0"})

    _rewrite(zipped, older)
    capsys.readouterr()
    assert main(["validate", str(zipped)]) == 0
    assert (
        f"warning: shoebox.zip was packed by acoustic_spaces 0.9.0, this is {__version__}"
        in capsys.readouterr().err
    )


def test_scene_files_include_array_elements(tmp_path: Path) -> None:
    _omni_csv(tmp_path / "element.csv")
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"] = {"box": [6.0, 4.0, 3.0]}
    data["outputs"] = [{"format": "binaural", "hrtf": "head.sofa"}]
    column = {"element": {"file": "element.csv"}, "count": 2, "spacing_m": 0.1}
    data["sources"][0]["directivity"] = {"array": column}
    scene = Scene.model_validate(data, context={"base_dir": tmp_path})
    assert scene.files() == [tmp_path / "element.csv", tmp_path / "head.sofa"]
    # Saving writes both relatively, as packing does.
    path = tmp_path / "scene.yaml"
    scene.to_yaml(path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["sources"][0]["directivity"]["array"]["element"]["file"] == "element.csv"
    assert saved["outputs"][0]["hrtf"] == "head.sofa"
    # The HRTF is not there to pack.
    with pytest.raises(ValueError, match="head.sofa, which is not there"):
        pack(scene, tmp_path / "out.zip", base_dir=tmp_path)
