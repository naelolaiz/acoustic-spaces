"""A scene packed in one file, to run on another computer: a server with a GPU, or a
job on a cloud service.

A pack is a zip file holding ``scene.yaml`` and every file the scene names (its model,
terrain, buildings map, material and directivity files, the sources' signals and HRTFs)
with the files those name in turn (a glTF model's buffers and images, a directivity
array's element data). Files under the scene's folder keep their place relative to it;
files elsewhere go under ``external/<their folder>/``. The scene file names them all
relatively, so the pack runs wherever it is unpacked, and every command that reads a
scene takes a pack in its place::

    acoustic-spaces pack hall/scene.yaml hall.zip
    acoustic-spaces simulate hall.zip --out results     # on the other computer

``pack.json`` records the version that wrote the pack and a SHA-256 of every file in it,
which unpacking checks. Packing uploads nothing: the pack is a file, to take where you
choose, and it holds everything needed to rebuild the scene.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from acoustic_spaces._version import __version__
from acoustic_spaces.scene import Scene

#: The scene file in a pack.
SCENE_FILE = "scene.yaml"
#: What the pack holds: its format, the version that wrote it and the files' hashes.
MANIFEST_FILE = "pack.json"
#: The folder of a pack that holds the files from outside the scene's folder.
EXTERNAL = "external"
#: The pack format this version writes and reads.
FORMAT = 1
#: Bytes read at a time from the files packed and unpacked (models and HRTFs can be
#: hundreds of megabytes).
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Packed:
    """A pack written by :func:`pack`."""

    path: Path
    #: (name in the pack, size in bytes) of every file of the scene in it.
    files: list[tuple[str, int]]

    @property
    def size(self) -> int:
        """The size of the scene's files in bytes, before compression."""
        return sum(size for _name, size in self.files)


@dataclass(frozen=True)
class Unpacked:
    """A pack unpacked by :func:`unpack`."""

    #: The scene file, ready to load.
    scene: Path
    #: The version of acoustic_spaces that wrote the pack.
    version: str


def is_pack(path: str | Path) -> bool:
    """Whether a path names a pack rather than a scene file: it ends in ``.zip``."""
    return Path(path).suffix.lower() == ".zip"


def pack(scene: Scene, out: str | Path, base_dir: str | Path | None = None) -> Packed:
    """Write ``scene`` and every file it names to the zip file ``out``.

    ``base_dir`` is the scene file's folder: the files under it keep their place
    relative to it. Without it (a scene never saved), every file goes under
    ``external/``.
    """
    out = Path(out)
    if not is_pack(out):
        raise ValueError(f"a pack is a zip file: name it {out.stem}.zip")
    base = None if base_dir is None else Path(base_dir).resolve()
    names, entries = _layout([Path(file).resolve() for file in scene.files()], base)
    text = scene.yaml_text(lambda file: names[Path(file).resolve()])
    hashes = {}
    try:
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(SCENE_FILE, text)
            for name, file in entries.items():
                archive.write(file, name)
                hashes[name] = _sha256(file)
            manifest = {
                "format": FORMAT,
                "acoustic_spaces": __version__,
                "scene": scene.name,
                "files": {SCENE_FILE: _sha256_of(text.encode("utf-8")), **hashes},
            }
            archive.writestr(MANIFEST_FILE, json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    return Packed(out, [(name, file.stat().st_size) for name, file in entries.items()])


def unpack(path: str | Path, into: str | Path) -> Unpacked:
    """Unpack the pack ``path`` into the folder ``into``, checking every file against
    the hash the pack records."""
    path, into = Path(path), Path(into)
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path.name} is not a pack: {exc}") from None
    with archive:
        try:
            manifest = json.loads(archive.read(MANIFEST_FILE))
        except KeyError:
            raise ValueError(f"{path.name} is not a pack: it has no {MANIFEST_FILE}") from None
        except ValueError as exc:
            raise ValueError(f"{path.name}: {MANIFEST_FILE} is not valid JSON: {exc}") from None
        if not isinstance(manifest, dict):
            raise ValueError(f"{path.name}: {MANIFEST_FILE} is not a pack's manifest")
        if manifest.get("format") != FORMAT:
            raise ValueError(
                f"{path.name} is a pack this version cannot read (it was written by "
                f"acoustic_spaces {manifest.get('acoustic_spaces', '?')}): update to that "
                "version or later"
            )
        files = manifest.get("files")
        if not isinstance(files, dict) or SCENE_FILE not in files:
            raise ValueError(f"{path.name}: {MANIFEST_FILE} lists no {SCENE_FILE}")
        for name, expected in files.items():
            target = into.joinpath(*_safe_parts(name, path))
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            try:
                with archive.open(name) as source, target.open("wb") as sink:
                    while chunk := source.read(_CHUNK):
                        digest.update(chunk)
                        sink.write(chunk)
            except KeyError:
                raise ValueError(f"{path.name} lacks {name}, which its manifest lists") from None
            except zipfile.BadZipFile:
                digest = None
            if digest is None or digest.hexdigest() != expected:
                raise ValueError(f"{path.name}: {name} is damaged (its hash does not match)")
    return Unpacked(into / SCENE_FILE, str(manifest.get("acoustic_spaces", "?")))


def _layout(files: list[Path], base: Path | None) -> tuple[dict[Path, str], dict[str, Path]]:
    """The name in the pack of each file the scene names, and every file to pack by its
    name. A file and the files it names keep their places relative to each other: under
    the scene's folder when they all lie in it, else in a folder of their own under
    ``external/`` named after the file's folder."""
    reserved = {SCENE_FILE, MANIFEST_FILE}
    inside, outside = {}, {}
    for file in files:
        if not file.is_file():
            raise ValueError(f"the scene names {file}, which is not there")
        group = [file, *_named_by(file)]
        missing = [f for f in group if not f.is_file()]
        if missing:
            raise ValueError(f"{file.name} names {missing[0]}, which is not there")
        if base is not None and all(f.is_relative_to(base) for f in group):
            relative = {f: f.relative_to(base).as_posix() for f in group}
            if not reserved & set(relative.values()):
                inside[file] = relative
                continue
        folder = file.parent
        try:
            outside[file] = {f: f.relative_to(folder).as_posix() for f in group}
        except ValueError:
            stray = next(f for f in group if not f.is_relative_to(folder))
            raise ValueError(
                f"{file.name} names {stray}, outside its folder: move that next to "
                f"{file.name} to pack it"
            ) from None
    names: dict[Path, str] = {}
    entries: dict[str, Path] = {}
    for file, relative in inside.items():
        names[file] = relative[file]
        entries.update((name, f) for f, name in relative.items())
    folders: dict[Path, str] = {}
    for file, relative in outside.items():
        if file.parent not in folders:
            label = file.parent.name or "files"
            prefix, count = f"{EXTERNAL}/{label}", 1
            while prefix in folders.values() or any(n.startswith(prefix + "/") for n in entries):
                count += 1
                prefix = f"{EXTERNAL}/{label}-{count}"
            folders[file.parent] = prefix
        prefix = folders[file.parent]
        names[file] = f"{prefix}/{relative[file]}"
        entries.update((f"{prefix}/{name}", f) for f, name in relative.items())
    return names, entries


def _named_by(file: Path) -> list[Path]:
    """The files that ``file`` names in turn, which must come with it: a glTF model's
    buffers and images, and the element's file of a directivity array (YAML, the only
    YAML file a scene names)."""
    suffix = file.suffix.lower()
    if suffix == ".gltf":
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read {file.name}: {exc}") from None
        uris = [item.get("uri") for key in ("buffers", "images") for item in data.get(key, [])]
        return [
            (file.parent / unquote(uri)).resolve()
            for uri in uris
            if isinstance(uri, str) and not uri.startswith("data:")
        ]
    if suffix in (".yaml", ".yml"):
        from acoustic_spaces.directivity import read_array

        file = getattr(read_array(file).element, "file", None)
        return [] if file is None else [Path(file).resolve()]
    return []


def _safe_parts(name: str, pack: Path) -> tuple[str, ...]:
    """The parts of a name in a pack, refusing any that would land outside the folder
    it is unpacked into."""
    parts = PurePosixPath(name).parts
    if (
        not parts
        or "\\" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") or ":" in part for part in parts)
    ):
        raise ValueError(f"{pack.name}: {name!r} is not a name a pack may hold")
    return parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
