"""Packs: a scene and every file it names in one zip, to run on another computer."""

import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
import yaml

from acoustic_spaces import __version__
from acoustic_spaces.audio import read_wav, write_wav
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


def _scene_playing(project: Path, signals: list[str]) -> Path:
    """A box whose sources play ``signals`` (paths from ``project``, written here)."""
    for i, signal in enumerate(signals):
        wav = (project / signal).resolve()
        wav.parent.mkdir(parents=True, exist_ok=True)
        write_wav(wav, np.full(480, 0.1 * (i + 1)), 48000)
    data = {
        "name": "Names",
        "geometry": {"box": [4.0, 3.0, 2.5]},
        "materials": {"m": {"absorption": [0.2] * 8}},
        "default_material": "m",
        "sources": [
            {"name": f"s{i}", "position": [1.0 + 0.2 * i, 1.0, 1.0], "signal": signal}
            for i, signal in enumerate(signals)
        ],
        "listeners": [{"name": "l", "position": [3.0, 2.0, 1.2]}],
    }
    path = project / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _apart(folder: Path, *names: str) -> bool:
    """Whether the file system of ``folder`` holds files of these names apart."""
    probe = folder / "names-probe"
    probe.mkdir()
    for name in names:
        (probe / name).touch()
    apart = len(list(probe.iterdir())) == len(names)
    shutil.rmtree(probe)
    return apart


@pytest.mark.skipif(sys.platform == "win32", reason="Windows holds no ':' in a file name")
def test_a_name_a_pack_cannot_hold_is_refused_when_packing(tmp_path: Path) -> None:
    """unpack refuses ':' and '\\' in a name on every system, so pack refuses such a
    name before it writes anything, naming the file. A folder of the packer's own
    naming (under external/) is given a name a pack holds."""
    project = tmp_path / "project"
    for signal in ("take 1: hall.wav", "back\\slash.wav"):
        path = _scene_playing(project, [signal])
        with pytest.raises(ValueError, match=r"cannot go in a pack as .*rename the file"):
            pack(Scene.from_yaml(path), tmp_path / "out.zip", base_dir=project)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["project"]
    path = _scene_playing(project, ["../take: 2/a.wav"])
    zipped = tmp_path / "out.zip"
    packed = pack(Scene.from_yaml(path), zipped, base_dir=project)
    assert [name for name, _bytes in packed.files] == ["external/take- 2/a.wav"]
    assert unpack(zipped, tmp_path / "unpacked").scene.is_file()


def test_names_that_differ_only_in_case_are_refused_or_kept_apart(tmp_path: Path) -> None:
    """Two files whose names differ only in case would be one file unpacked on Windows
    or macOS, and a source would play the other's signal: pack refuses them, naming
    both. Folders of the packer's naming are kept apart where case does not count."""
    if not _apart(tmp_path, "Take.wav", "take.wav"):
        pytest.skip("this file system holds no two names that differ only in case")
    project = tmp_path / "project"
    path = _scene_playing(project, ["Take.wav", "take.wav"])
    where = "only in case: unpacked on Windows or macOS, one would take the other's place"
    with pytest.raises(ValueError, match=rf"Take.wav and .*take.wav would go .*{where}"):
        pack(Scene.from_yaml(path), tmp_path / "out.zip", base_dir=project)
    assert not (tmp_path / "out.zip").exists()
    path = _scene_playing(project, ["../Snd/a.wav", "../snd/a.wav", "PACK.json"])
    zipped = tmp_path / "out.zip"
    packed = pack(Scene.from_yaml(path), zipped, base_dir=project)
    assert sorted(name for name, _bytes in packed.files) == [
        "external/Snd/a.wav",
        "external/project/PACK.json",
        "external/snd-2/a.wav",
    ]
    unpacked = Room.from_yaml(unpack(zipped, tmp_path / "unpacked").scene)
    signals = [read_wav(source.signal)[0][0, 0] for source in unpacked.scene.sources]
    np.testing.assert_allclose(signals, [0.1, 0.2, 0.3], atol=1e-4)


ENCODED = "the way a letter is encoded (é as one character, or as e and an accent)"


@pytest.mark.parametrize(
    ("names", "difference"),
    [
        # é as one character (NFC), and as e and a combining accent (NFD).
        (["\u00e9.wav", "e\u0301.wav"], ENCODED),
        # Full case folding: ß is ss. NTFS upper-cases ß to itself.
        (["stra\u00dfe.wav", "strasse.wav"], "case"),
        # ΐ, and Ϊ with an accent: their folds are one name once put in NFD again.
        (["\u0390.wav", "\u03aa\u0301.wav"], f"case and in {ENCODED}"),
    ],
    ids=["encoding", "sharp-s", "case-and-encoding"],
)
def test_names_macos_takes_for_one_are_refused_saying_how(tmp_path, names, difference) -> None:
    """APFS takes names of one canonical caseless form (The Unicode Standard, D145: NFD
    after full case folding, then NFD) for one, where NTFS keeps these apart: pack
    refuses them, saying what they differ in and that macOS would unpack them as one."""
    if not _apart(tmp_path, *names):
        pytest.skip("this file system takes these names for one")
    project = tmp_path / "project"
    path = _scene_playing(project, names)
    with pytest.raises(ValueError) as error:
        pack(Scene.from_yaml(path), tmp_path / "out.zip", base_dir=project)
    first, second = (project.resolve() / name for name in names)
    assert str(error.value) == (
        f"{first} and {second} would go in a pack as {names[0]!r} and {names[1]!r}, which "
        f"differ only in {difference}: unpacked on macOS, one would take the other's "
        "place; rename one"
    )
    assert not (tmp_path / "out.zip").exists()


def test_a_folder_of_the_packers_naming_takes_no_files_name(tmp_path: Path) -> None:
    """A folder of the packer's naming under external/ takes no file's name there:
    unpacked, the second could not be written on any system."""
    project = tmp_path / "project"
    path = _scene_playing(project, ["external/a.wav", "../a.wav/x.wav"])
    zipped = tmp_path / "out.zip"
    packed = pack(Scene.from_yaml(path), zipped, base_dir=project)
    assert sorted(name for name, _bytes in packed.files) == [
        "external/a.wav",
        "external/a.wav-2/x.wav",
    ]
    unpacked = Room.from_yaml(unpack(zipped, tmp_path / "unpacked").scene)
    signals = [read_wav(source.signal)[0][0, 0] for source in unpacked.scene.sources]
    np.testing.assert_allclose(signals, [0.1, 0.2], atol=1e-4)


def test_a_file_and_a_folder_whose_names_differ_only_in_case_are_refused(tmp_path) -> None:
    """A file a and a folder A cannot be unpacked side by side on Windows or macOS: pack
    refuses them, naming both."""
    if not _apart(tmp_path, "a", "A"):
        pytest.skip("this file system holds no two names that differ only in case")
    project, zipped = tmp_path / "project", tmp_path / "out.zip"
    path = _scene_playing(project, ["a", "A/b.wav"])
    with pytest.raises(ValueError) as error:
        pack(Scene.from_yaml(path), zipped, base_dir=project)
    assert str(error.value) == (
        f"{project.resolve() / 'a'} would go in a pack as 'a', and "
        f"{project.resolve() / 'A' / 'b.wav'} as "
        "'A/b.wav', in a folder 'A', a name that differs from the file's only in case: "
        "unpacked on Windows or macOS, a file and a folder would have one name; rename one"
    )
    assert not zipped.exists()


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


def test_pack_help_lists_what_a_pack_holds(capsys) -> None:
    """pack --help names every kind of file Scene.files() puts in a pack, the sources'
    signals and the scene's recording among them."""
    with pytest.raises(SystemExit):
        main(["pack", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    for kind in ("model", "terrain", "buildings map", "material", "directivity"):
        assert kind in text
    assert "the files sources play, the scene's recording and HRTFs" in text


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
    # Checked before it takes its name: no damaged file is left.
    assert not (tmp_path / "d" / "hill.asc").exists()
    assert not list((tmp_path / "d").rglob("*.part"))
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


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [(8, 1, "is encrypted"), (10, 99, "compression method is not supported")],
)
def test_unpack_refuses_a_file_it_cannot_read(tmp_path: Path, offset, value, message) -> None:
    # An encrypted file, or one compressed some other way, raised zipfile's own error.
    zipped = tmp_path / "hillside.zip"
    pack(Scene.from_yaml(EXAMPLES / "hillside" / "scene.yaml"), zipped, EXAMPLES / "hillside")
    data = bytearray(zipped.read_bytes())
    entry = data.find(b"PK\x01\x02")  # the first file's record in the central directory
    struct.pack_into("<H", data, entry + offset, value)
    zipped.write_bytes(bytes(data))
    first = zipfile.ZipFile(zipped).namelist()[0]
    with pytest.raises(ValueError, match=f"{first} is damaged .*{message}"):
        unpack(zipped, tmp_path / "u")
    assert not list((tmp_path / "u").rglob("*.part"))


def test_unpack_needs_room_for_the_files(tmp_path: Path, monkeypatch) -> None:
    zipped = tmp_path / "hillside.zip"
    packed = pack(
        Scene.from_yaml(EXAMPLES / "hillside" / "scene.yaml"), zipped, EXAMPLES / "hillside"
    )
    free = packed.size // 2
    monkeypatch.setattr(
        "acoustic_spaces.pack.shutil.disk_usage", lambda path: SimpleNamespace(free=free)
    )
    with pytest.raises(ValueError, match="MiB free"):
        unpack(zipped, tmp_path / "u")
    assert not any((tmp_path / "u").iterdir())
